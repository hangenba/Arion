package bert

import (
	"Arion/configs"
	"Arion/pkg/btp"
	"Arion/pkg/ecdmat"
	"Arion/pkg/he"
	"Arion/pkg/math/activation/gelu"
	"Arion/pkg/math/activation/layernorm"
	"Arion/pkg/math/activation/tanh"
	"Arion/pkg/math/matrix"
	"Arion/pkg/utils"
	"fmt"
	"runtime"
	"time"
)

// InferenceBert performs the BERT model inference under the CKKS scheme.
// It initializes the model parameters, reads input data, encrypts it, and performs the inference
// through multiple layers of attention mechanisms.
// paramType specifies the type of parameters to use, e.g., "short", "base".
// It prints the progress and results of each layer's attention computation.
// Note: This function assumes that the necessary packages and configurations are correctly set up.
func InferenceBertMT(paramType string, rowValue int, modelPath string) {

	prof := newProfiler() // ← 新建统计器

	totalStart := time.Now() // === 统计总时间 ===

	fmt.Print("----------Task: BERT-Tiny Model Inference Under The CKKS Scheme " + paramType + "----------\n")

	/* Group1: Generate the model parameters, configurations and data */
	ckksParams, btpParams, modelParams, err := configs.InitBert(paramType, rowValue, modelPath)
	if err != nil {
		panic(err)
	}

	numThreads := 64
	runtime.GOMAXPROCS(numThreads)
	fmt.Printf("GOMAXPROCS is set to: %d\n", runtime.GOMAXPROCS(0))

	outputFilePath := "Output/" + modelParams.ModelPath

	mat, err := utils.ReadInput(modelParams, "embedded_inputs.csv")
	if err != nil {
		panic("failed to read input: " + err.Error())
	}
	mat = utils.PadOrTruncateMatrix(mat, modelParams)
	inputMats := utils.MatrixToBatchMats(mat, modelParams)

	// === 密钥生成耗时统计 ===
	keyGenStart := time.Now()
	ecd, enc, eval, dec, btpEval := he.GenerateKeysAndBtsKeysMT(ckksParams, btpParams, modelParams, numThreads)
	keyGenElapsed := time.Since(keyGenStart)
	fmt.Printf("Key Generation & Bootstrapping Keys Time: %s\n", keyGenElapsed)

	// === 输入加密耗时统计 ===
	encryptStart := time.Now()
	encodedInputs := ecdmat.EncodeDense(inputMats, modelParams)
	ctX, err := he.EncryptInputMatricesMT(encodedInputs, modelParams, ckksParams, enc, ecd, numThreads)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt input matrices: %v", err))
	}
	encryptElapsed := time.Since(encryptStart)
	fmt.Printf("Input Encryption Time: %s\n", encryptElapsed)

	// === 逐层推理 ===
	for i := 0; i < modelParams.NumLayers; i++ {
		fmt.Printf("================== Layer %d  ==================\n", i)

		layerStart := time.Now() // === 每层开始时间 ===

		// 读取参数
		layerParams, err := utils.ReadLayerParameters(modelParams, i)
		if err != nil {
			panic(fmt.Sprintf("failed to read layer parameters for layer %d: %v", i, err))
		}

		// Compute QKV
		stop := prof.Enter("Attention/Pt-ct MatMul")
		ctQ, ctK, ctV, err := ComputeQKVMT(ctX, layerParams, modelParams, ckksParams, eval, numThreads)
		if err != nil {
			panic(fmt.Sprintf("failed to compute QKV matrices: %v", err))
		}
		stop() // 结束计时

		ptQDec, _ := he.DecryptCiphertextMatricesMT(ctQ, modelParams, ckksParams, dec, ecd, numThreads)
		ptQDecode := ecdmat.DecodeDense(ptQDec, modelParams)
		fmt.Println("Compute QKV finished")
		utils.PrintMat(ptQDecode[0])

		// MultiHead Attention
		stop = prof.Enter("Attention/MultiHead")
		ctAttention, err := ComputeMultiHeadAttentionMT1(ctQ, ctK, ctV, modelParams, &ckksParams, ecd, enc, eval, btpEval, numThreads)
		if err != nil {
			panic(err)
		}
		stop() // 结束计时
		fmt.Println("Attention computation finished")

		ptAttentionDec, _ := he.DecryptCiphertextMatricesMT(ctAttention, modelParams, ckksParams, dec, ecd, numThreads)
		ptAttentionDecode := ecdmat.DecodeDense(ptAttentionDec, modelParams)
		utils.PrintMat(ptAttentionDecode[0])

		// Bootstrapping 1
		stop = prof.Enter("Attention/Bootstrap-1")
		ctAttentionOutput, err := btp.CiphertextMatricesBootstrappingMT(ctAttention, btpEval, btpParams, numThreads)
		if err != nil {
			panic(fmt.Sprintf("failed to bootstrap attention ciphertexts: %v", err))
		}
		fmt.Println("Bootstrapping for attention finished")
		stop() // 结束计时

		ptAttentionBtsDec, _ := he.DecryptCiphertextMatricesMT(ctAttentionOutput, modelParams, ckksParams, dec, ecd, numThreads)
		ptAttentionBtsDecode := ecdmat.DecodeDense(ptAttentionBtsDec, modelParams)
		utils.PrintMat(ptAttentionBtsDecode[0])

		// SelfOutput + Residual + LayerNorm
		stop = prof.Enter("Self/Pt-ct MatMul")
		ctSelfOutputDense, _ := matrix.CiphertextMatricesMultiplyWeightAndAddBiasMT(ctAttentionOutput, layerParams.LayerAttentionOutputDenseWeight.T(), layerParams.LayerAttentionOutputDenseBias, modelParams, ckksParams, eval, numThreads)
		// fmt.Printf("Dense MatMul Scale: %f\n", &ctSelfOutputDense.Ciphertexts[0].Scale.Value)
		// fmt.Printf("ctX MatMul Scale: %f\n", &ctX.Ciphertexts[0].Scale.Value)
		ctSelfOutputResidual, _ := matrix.CiphertextMatricesAddCiphertextMatricesMT(ctSelfOutputDense, ctX, eval, numThreads)
		stop() // 结束计时

		fmt.Println("Self/Pt-ct MatMul finished")
		ptSelfOutputDenseDec, _ := he.DecryptCiphertextMatricesMT(ctSelfOutputDense, modelParams, ckksParams, dec, ecd, numThreads)
		ptSelfOutputDenseDecode := ecdmat.DecodeDense(ptSelfOutputDenseDec, modelParams)
		utils.PrintMat(ptSelfOutputDenseDecode[0])

		stop = prof.Enter("Self/LayerNorm")
		ctSelfOutputLayernorm, _ := layernorm.LayerNormSelfAttentionOutputMT(ctSelfOutputResidual, layerParams.LayerAttentionOutputLayerNormWeight, layerParams.LayerAttentionOutputLayerNormBias, modelParams, ckksParams, ecd, eval, numThreads)
		fmt.Println("SelfOutputLayernorm computation finished")
		stop() // 结束计时

		ptSelfOutputDec, _ := he.DecryptCiphertextMatricesMT(ctSelfOutputLayernorm, modelParams, ckksParams, dec, ecd, numThreads)
		ptSelfOutputDecode := ecdmat.DecodeDense(ptSelfOutputDec, modelParams)
		utils.PrintMat(ptSelfOutputDecode[0])

		// Bootstrapping 2
		stop = prof.Enter("Self/Bootstrap-2")
		ctSelfOutput, _ := btp.CiphertextMatricesBootstrappingMT(ctSelfOutputLayernorm, btpEval, btpParams, numThreads)
		fmt.Println("Bootstrapping for SelfOutputLayernorm finished")
		stop() // 结束计时

		// Intermediate
		stop = prof.Enter("Inter/Pt-ct-1 MatMul")
		ctIntermediate, _ := matrix.CiphertextMatricesMultiplyWeightAndAddBiasMT(ctSelfOutput, layerParams.LayerIntermediateDenseWeight.T(), layerParams.LayerIntermediateDenseBias, modelParams, ckksParams, eval, numThreads)
		// fmt.Printf("ctSelfOutput MatMul Scale: %f\n", &ctSelfOutput.Ciphertexts[0].Scale.Value)
		// fmt.Printf("ctIntermediate MatMul Scale: %f\n", &ctIntermediate.Ciphertexts[0].Scale.Value)
		stop() // 结束计时

		stop = prof.Enter("Inter/GELU")
		ctGeLU := gelu.CiphertextMatricesGeluChebyshevMT(ctIntermediate, &ckksParams, eval, modelParams.GeluMinValue, modelParams.GeluMaxValue, modelParams.GeluDegree, numThreads)
		stop() // 结束计时

		stop = prof.Enter("Inter/Pt-ct-2 MatMul")
		ctOutputDense, _ := matrix.CiphertextMatricesMultiplyWeightAndAddBiasMT(ctGeLU, layerParams.LayerOutputDenseWeight.T(), layerParams.LayerOutputDenseBias, modelParams, ckksParams, eval, numThreads)
		// fmt.Printf("ctGeLU MatMul Scale: %f\n", &ctGeLU.Ciphertexts[0].Scale.Value)
		// fmt.Printf("ctOutputDense MatMul Scale: %f\n", &ctOutputDense.Ciphertexts[0].Scale.Value)
		stop() // 结束计时

		fmt.Println("Intermediate computation finished")
		ptOutputDenseDec, _ := he.DecryptCiphertextMatricesMT(ctOutputDense, modelParams, ckksParams, dec, ecd, numThreads)
		ptOutputDenseDecode := ecdmat.DecodeDense(ptOutputDenseDec, modelParams)
		utils.PrintMat(ptOutputDenseDecode[0])

		// Bootstrapping 3
		stop = prof.Enter("Inter/Bootstrap-3")
		ctOutputLayernormInput, _ := btp.CiphertextMatricesBootstrappingMT(ctOutputDense, btpEval, btpParams, numThreads)
		stop() // 结束计时
		fmt.Println("Bootstrapping for Intermediate finished")

		// Residual + LayerNorm
		stop = prof.Enter("Final/LayerNorm")
		ctOutputResidual, _ := matrix.CiphertextMatricesAddCiphertextMatricesMT(ctOutputLayernormInput, ctSelfOutput, eval, numThreads)
		ctOutputLayernorm, _ := layernorm.LayerNormOutputMT(ctOutputResidual, layerParams.LayerOutputLayerNormWeight, layerParams.LayerOutputLayerNormBias, modelParams, ckksParams, ecd, eval, numThreads)
		// fmt.Printf("ctOutputResidual MatMul Scale: %f\n", &ctOutputResidual.Ciphertexts[0].Scale.Value)
		// fmt.Printf("ctOutputLayernorm MatMul Scale: %f\n", &ctOutputLayernorm.Ciphertexts[0].Scale.Value)
		stop() // 结束计时
		fmt.Println("OutputLayernorm computation finished")

		// Bootstrapping 4
		stop = prof.Enter("Final/Bootstrap-4")
		ctOutput, _ := btp.CiphertextMatricesBootstrappingMT(ctOutputLayernorm, btpEval, btpParams, numThreads)
		stop() // 结束计时
		fmt.Println("Bootstrapping for OutputLayernorm finished")

		// 解密输出
		ptDec, _ := he.DecryptCiphertextMatricesMT(ctOutput, modelParams, ckksParams, dec, ecd, numThreads)
		ptDecode := ecdmat.DecodeDense(ptDec, modelParams)
		utils.PrintMat(ptDecode[0])
		filePath := fmt.Sprintf(outputFilePath+"/layer_%d_output.csv", i)
		_ = utils.SaveMatrixToCSV(ptDecode[0], filePath)

		ctX = he.DeepCopyCiphertextMatrices(ctOutput)

		layerElapsed := time.Since(layerStart)
		fmt.Printf("Layer %d Computation Time: %s\n", i, layerElapsed)
	}

	prof.Report(modelParams.NumBatch)
	fmt.Printf("==========   Wall-Clock Total: %s   ==========\n", time.Since(totalStart))
}

func InferenceBertTinyMT(paramType string, rowValue int, modelPath string) {

	prof := newProfiler() // ← 新建统计器

	totalStart := time.Now() // === 统计总时间 ===

	fmt.Print("----------Task: BERT-Tiny Model Inference Under The CKKS Scheme " + paramType + "----------\n")

	/* Group1: Generate the model parameters, configurations and data */
	ckksParams, btpParams, modelParams, err := configs.InitBertTiny(paramType, rowValue, modelPath)
	if err != nil {
		panic(err)
	}

	numThreads := 64
	runtime.GOMAXPROCS(numThreads)
	fmt.Printf("GOMAXPROCS is set to: %d\n", runtime.GOMAXPROCS(0))

	outputFilePath := "Output/" + modelParams.ModelPath

	mat, err := utils.ReadInput(modelParams, "embedded_inputs.csv")
	if err != nil {
		panic("failed to read input: " + err.Error())
	}
	mat = utils.PadOrTruncateMatrix(mat, modelParams)
	inputMats := utils.MatrixToBatchMats(mat, modelParams)

	// === 密钥生成耗时统计 ===
	keyGenStart := time.Now()
	ecd, enc, eval, dec, btpEval := he.GenerateKeysAndBtsKeysMT(ckksParams, btpParams, modelParams, numThreads)
	keyGenElapsed := time.Since(keyGenStart)
	fmt.Printf("Key Generation & Bootstrapping Keys Time: %s\n", keyGenElapsed)

	// === 输入加密耗时统计 ===
	encryptStart := time.Now()
	encodedInputs := ecdmat.EncodeDense(inputMats, modelParams)
	ctX, err := he.EncryptInputMatricesMT(encodedInputs, modelParams, ckksParams, enc, ecd, numThreads)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt input matrices: %v", err))
	}
	encryptElapsed := time.Since(encryptStart)
	fmt.Printf("Input Encryption Time: %s\n", encryptElapsed)

	// === 逐层推理 ===
	for i := 0; i < modelParams.NumLayers; i++ {
		fmt.Printf("================== Layer %d  ==================\n", i)

		layerStart := time.Now() // === 每层开始时间 ===

		// 读取参数
		layerParams, err := utils.ReadLayerParameters(modelParams, i)
		if err != nil {
			panic(fmt.Sprintf("failed to read layer parameters for layer %d: %v", i, err))
		}

		// Compute QKV
		stop := prof.Enter("Attention/Pt-ct MatMul")
		ctQ, ctK, ctV, err := ComputeQKVMT(ctX, layerParams, modelParams, ckksParams, eval, numThreads)
		if err != nil {
			panic(fmt.Sprintf("failed to compute QKV matrices: %v", err))
		}
		stop() // 结束计时

		// MultiHead Attention
		stop = prof.Enter("Attention/MultiHead")
		ctAttention, err := ComputeMultiHeadAttentionMT1(ctQ, ctK, ctV, modelParams, &ckksParams, ecd, enc, eval, btpEval, numThreads)
		if err != nil {
			panic(err)
		}
		stop() // 结束计时
		fmt.Println("Attention computation finished")

		// Bootstrapping 1
		stop = prof.Enter("Attention/Bootstrap-1")
		ctAttentionOutput, err := btp.CiphertextMatricesBootstrappingMT(ctAttention, btpEval, btpParams, numThreads)
		if err != nil {
			panic(fmt.Sprintf("failed to bootstrap attention ciphertexts: %v", err))
		}
		fmt.Println("Bootstrapping for attention finished")
		stop() // 结束计时

		// SelfOutput + Residual + LayerNorm
		stop = prof.Enter("Self/Pt-ct MatMul")
		ctSelfOutputDense, _ := matrix.CiphertextMatricesMultiplyWeightAndAddBiasMT(ctAttentionOutput, layerParams.LayerAttentionOutputDenseWeight.T(), layerParams.LayerAttentionOutputDenseBias, modelParams, ckksParams, eval, numThreads)
		ctSelfOutputResidual, _ := matrix.CiphertextMatricesAddCiphertextMatricesMT(ctSelfOutputDense, ctX, eval, numThreads)
		stop() // 结束计时

		stop = prof.Enter("Self/LayerNorm")
		ctSelfOutputLayernorm, _ := layernorm.LayerNormSelfAttentionOutputMT(ctSelfOutputResidual, layerParams.LayerAttentionOutputLayerNormWeight, layerParams.LayerAttentionOutputLayerNormBias, modelParams, ckksParams, ecd, eval, numThreads)
		fmt.Println("SelfOutputLayernorm computation finished")
		stop() // 结束计时

		// Bootstrapping 2
		stop = prof.Enter("Self/Bootstrap-2")
		ctSelfOutput, _ := btp.CiphertextMatricesBootstrappingMT(ctSelfOutputLayernorm, btpEval, btpParams, numThreads)
		fmt.Println("Bootstrapping for SelfOutputLayernorm finished")
		stop() // 结束计时

		// Intermediate
		stop = prof.Enter("Inter/Pt-ct-1 MatMul")
		ctIntermediate, _ := matrix.CiphertextMatricesMultiplyWeightAndAddBiasMT(ctSelfOutput, layerParams.LayerIntermediateDenseWeight.T(), layerParams.LayerIntermediateDenseBias, modelParams, ckksParams, eval, numThreads)
		stop() // 结束计时

		stop = prof.Enter("Inter/GELU")
		ctGeLU := gelu.CiphertextMatricesGeluChebyshevMT(ctIntermediate, &ckksParams, eval, modelParams.GeluMinValue, modelParams.GeluMaxValue, modelParams.GeluDegree, numThreads)
		stop() // 结束计时

		stop = prof.Enter("Inter/Pt-ct-2 MatMul")
		ctOutputDense, _ := matrix.CiphertextMatricesMultiplyWeightAndAddBiasMT(ctGeLU, layerParams.LayerOutputDenseWeight.T(), layerParams.LayerOutputDenseBias, modelParams, ckksParams, eval, numThreads)
		stop() // 结束计时

		fmt.Println("Intermediate computation finished")

		// Bootstrapping 3
		stop = prof.Enter("Inter/Bootstrap-3")
		ctOutputLayernormInput, _ := btp.CiphertextMatricesBootstrappingMT(ctOutputDense, btpEval, btpParams, numThreads)
		stop() // 结束计时
		fmt.Println("Bootstrapping for Intermediate finished")

		// Residual + LayerNorm
		stop = prof.Enter("Final/LayerNorm")
		ctOutputResidual, _ := matrix.CiphertextMatricesAddCiphertextMatricesMT(ctOutputLayernormInput, ctSelfOutput, eval, numThreads)
		ctOutputLayernorm, _ := layernorm.LayerNormOutputMT(ctOutputResidual, layerParams.LayerOutputLayerNormWeight, layerParams.LayerOutputLayerNormBias, modelParams, ckksParams, ecd, eval, numThreads)
		stop() // 结束计时
		fmt.Println("OutputLayernorm computation finished")

		// Bootstrapping 4
		stop = prof.Enter("Final/Bootstrap-4")
		ctOutput, _ := btp.CiphertextMatricesBootstrappingMT(ctOutputLayernorm, btpEval, btpParams, numThreads)
		stop() // 结束计时
		fmt.Println("Bootstrapping for OutputLayernorm finished")

		// 解密输出
		ptDec, _ := he.DecryptCiphertextMatricesMT(ctOutput, modelParams, ckksParams, dec, ecd, numThreads)
		ptDecode := ecdmat.DecodeDense(ptDec, modelParams)
		utils.PrintMat(ptDecode[0])
		filePath := fmt.Sprintf(outputFilePath+"/layer_%d_output.csv", i)
		_ = utils.SaveMatrixToCSV(ptDecode[0], filePath)

		ctX = he.DeepCopyCiphertextMatrices(ctOutput)

		layerElapsed := time.Since(layerStart)
		fmt.Printf("Layer %d Computation Time: %s\n", i, layerElapsed)
	}

	fmt.Printf("================== Pooler  ==================\n")
	// Pooler Output (同理可加时间统计)
	poolerStart := time.Now()

	// Read Pooler Parameters
	poolerDenseWeight, poolerDenseBias, err := utils.ReadPoolerParameters(modelParams)
	if err != nil {
		panic(fmt.Sprintf("failed to read pooler parameters: %v", err))
	}

	// Linear Transformation
	poolerLinearOutput, err := matrix.CiphertextMatricesMultiplyWeightAndAddBiasMT(ctX, poolerDenseWeight.T(), poolerDenseBias, modelParams, ckksParams, eval, numThreads)
	if err != nil {
		panic(err)
	}

	// Tanh Activation
	poolerLinearBeforeTanhOutput, err := tanh.CiphertextMatricesOutputLinear(poolerLinearOutput, modelParams, ckksParams, eval)
	if err != nil {
		panic(err)
	}
	poolerTanhOutput := tanh.CiphertextMatricesTanhChebyshevMT(poolerLinearBeforeTanhOutput, &ckksParams, eval, numThreads)

	// Decrypt and Decode Pooler Output
	ptPoolerOutput, err := he.DecryptCiphertextMatricesMT(poolerTanhOutput, modelParams, ckksParams, dec, ecd, numThreads)
	if err != nil {
		panic(err)
	}
	ptPoolerDecode := ecdmat.DecodeOutputDenseToMatrix(ptPoolerOutput, modelParams)
	fmt.Println("Pooler output:")
	utils.PrintMat(ptPoolerDecode)

	// Save Pooler Output to CSV
	err = utils.SaveMatrixToCSV(ptPoolerDecode, outputFilePath+"/bert_output.csv")
	if err != nil {
		fmt.Println("保存失败:", err)
	}

	poolerElapsed := time.Since(poolerStart)
	fmt.Printf("Pooler Computation Time: %s\n", poolerElapsed)

	prof.Report(modelParams.NumBatch)
	fmt.Printf("==========   Wall-Clock Total: %s   ==========\n", time.Since(totalStart))
}

func InferenceBertTinyClassifierMT(paramType string, rowValue int, modelPath string) {

	prof := newProfiler() // ← 新建统计器

	totalStart := time.Now() // === 统计总时间 ===

	fmt.Print("----------Task: BERT-Tiny Model Inference Under The CKKS Scheme " + paramType + "----------\n")

	/* Group1: Generate the model parameters, configurations and data */
	ckksParams, btpParams, modelParams, err := configs.InitBertTiny(paramType, rowValue, modelPath)
	if err != nil {
		panic(err)
	}

	numThreads := 64
	runtime.GOMAXPROCS(numThreads)
	fmt.Printf("GOMAXPROCS is set to: %d\n", runtime.GOMAXPROCS(0))

	outputFilePath := "Output/" + modelParams.ModelPath

	mat, err := utils.ReadInput(modelParams, "embedded_inputs.csv")
	if err != nil {
		panic("failed to read input: " + err.Error())
	}
	mat = utils.PadOrTruncateMatrix(mat, modelParams)
	inputMats := utils.MatrixToBatchMats(mat, modelParams)

	// === 密钥生成耗时统计 ===
	keyGenStart := time.Now()
	ecd, enc, eval, dec, btpEval := he.GenerateKeysAndBtsKeysMT(ckksParams, btpParams, modelParams, numThreads)
	keyGenElapsed := time.Since(keyGenStart)
	fmt.Printf("Key Generation & Bootstrapping Keys Time: %s\n", keyGenElapsed)

	// === 输入加密耗时统计 ===
	encryptStart := time.Now()
	encodedInputs := ecdmat.EncodeDense(inputMats, modelParams)
	ctX, err := he.EncryptInputMatricesMT(encodedInputs, modelParams, ckksParams, enc, ecd, numThreads)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt input matrices: %v", err))
	}
	encryptElapsed := time.Since(encryptStart)
	fmt.Printf("Input Encryption Time: %s\n", encryptElapsed)

	// === 逐层推理 ===
	for i := 0; i < modelParams.NumLayers; i++ {
		fmt.Printf("================== Layer %d  ==================\n", i)

		layerStart := time.Now() // === 每层开始时间 ===

		// 读取参数
		layerParams, err := utils.ReadLayerParameters(modelParams, i)
		if err != nil {
			panic(fmt.Sprintf("failed to read layer parameters for layer %d: %v", i, err))
		}

		// Compute QKV
		stop := prof.Enter("Attention/Pt-ct MatMul")
		ctQ, ctK, ctV, err := ComputeQKVMT(ctX, layerParams, modelParams, ckksParams, eval, numThreads)
		if err != nil {
			panic(fmt.Sprintf("failed to compute QKV matrices: %v", err))
		}
		stop() // 结束计时

		ptQDec, _ := he.DecryptCiphertextMatricesMT(ctQ, modelParams, ckksParams, dec, ecd, numThreads)
		ptQDecode := ecdmat.DecodeDense(ptQDec, modelParams)
		utils.PrintMat(ptQDecode[0])

		// MultiHead Attention
		stop = prof.Enter("Attention/MultiHead")
		ctAttention, err := ComputeMultiHeadAttentionMT1(ctQ, ctK, ctV, modelParams, &ckksParams, ecd, enc, eval, btpEval, numThreads)
		if err != nil {
			panic(err)
		}
		stop() // 结束计时
		fmt.Println("Attention computation finished")

		ptAttentionDec, _ := he.DecryptCiphertextMatricesMT(ctAttention, modelParams, ckksParams, dec, ecd, numThreads)
		ptAttentionDecode := ecdmat.DecodeDense(ptAttentionDec, modelParams)
		utils.PrintMat(ptAttentionDecode[0])

		// Bootstrapping 1
		stop = prof.Enter("Attention/Bootstrap-1")
		ctAttentionOutput, err := btp.CiphertextMatricesBootstrappingMT(ctAttention, btpEval, btpParams, numThreads)
		if err != nil {
			panic(fmt.Sprintf("failed to bootstrap attention ciphertexts: %v", err))
		}
		fmt.Println("Bootstrapping for attention finished")
		stop() // 结束计时

		// SelfOutput + Residual + LayerNorm
		stop = prof.Enter("Self/Pt-ct MatMul")
		ctSelfOutputDense, _ := matrix.CiphertextMatricesMultiplyWeightAndAddBiasMT(ctAttentionOutput, layerParams.LayerAttentionOutputDenseWeight.T(), layerParams.LayerAttentionOutputDenseBias, modelParams, ckksParams, eval, numThreads)
		ctSelfOutputResidual, _ := matrix.CiphertextMatricesAddCiphertextMatricesMT(ctSelfOutputDense, ctX, eval, numThreads)
		stop() // 结束计时

		stop = prof.Enter("Self/LayerNorm")
		ctSelfOutputLayernorm, _ := layernorm.LayerNormSelfAttentionOutputMT(ctSelfOutputResidual, layerParams.LayerAttentionOutputLayerNormWeight, layerParams.LayerAttentionOutputLayerNormBias, modelParams, ckksParams, ecd, eval, numThreads)
		fmt.Println("SelfOutputLayernorm computation finished")
		stop() // 结束计时

		ptSelfOutputDec, _ := he.DecryptCiphertextMatricesMT(ctSelfOutputLayernorm, modelParams, ckksParams, dec, ecd, numThreads)
		ptSelfOutputDecode := ecdmat.DecodeDense(ptSelfOutputDec, modelParams)
		utils.PrintMat(ptSelfOutputDecode[0])

		// Bootstrapping 2
		stop = prof.Enter("Self/Bootstrap-2")
		ctSelfOutput, _ := btp.CiphertextMatricesBootstrappingMT(ctSelfOutputLayernorm, btpEval, btpParams, numThreads)
		fmt.Println("Bootstrapping for SelfOutputLayernorm finished")
		stop() // 结束计时

		// Intermediate
		stop = prof.Enter("Inter/Pt-ct-1 MatMul")
		ctIntermediate, _ := matrix.CiphertextMatricesMultiplyWeightAndAddBiasMT(ctSelfOutput, layerParams.LayerIntermediateDenseWeight.T(), layerParams.LayerIntermediateDenseBias, modelParams, ckksParams, eval, numThreads)
		stop() // 结束计时

		stop = prof.Enter("Inter/GELU")
		ctGeLU := gelu.CiphertextMatricesGeluChebyshevMT(ctIntermediate, &ckksParams, eval, modelParams.GeluMinValue, modelParams.GeluMaxValue, modelParams.GeluDegree, numThreads)
		stop() // 结束计时

		stop = prof.Enter("Inter/Pt-ct-2 MatMul")
		ctOutputDense, _ := matrix.CiphertextMatricesMultiplyWeightAndAddBiasMT(ctGeLU, layerParams.LayerOutputDenseWeight.T(), layerParams.LayerOutputDenseBias, modelParams, ckksParams, eval, numThreads)
		stop() // 结束计时

		fmt.Println("Intermediate computation finished")

		ptOutputDenseDec, _ := he.DecryptCiphertextMatricesMT(ctOutputDense, modelParams, ckksParams, dec, ecd, numThreads)
		ptOutputDenseDecode := ecdmat.DecodeDense(ptOutputDenseDec, modelParams)
		utils.PrintMat(ptOutputDenseDecode[0])

		// Bootstrapping 3
		stop = prof.Enter("Inter/Bootstrap-3")
		ctOutputLayernormInput, _ := btp.CiphertextMatricesBootstrappingMT(ctOutputDense, btpEval, btpParams, numThreads)
		stop() // 结束计时
		fmt.Println("Bootstrapping for Intermediate finished")

		// Residual + LayerNorm
		stop = prof.Enter("Final/LayerNorm")
		ctOutputResidual, _ := matrix.CiphertextMatricesAddCiphertextMatricesMT(ctOutputLayernormInput, ctSelfOutput, eval, numThreads)
		ctOutputLayernorm, _ := layernorm.LayerNormOutputMT(ctOutputResidual, layerParams.LayerOutputLayerNormWeight, layerParams.LayerOutputLayerNormBias, modelParams, ckksParams, ecd, eval, numThreads)
		stop() // 结束计时
		fmt.Println("OutputLayernorm computation finished")

		// Bootstrapping 4
		stop = prof.Enter("Final/Bootstrap-4")
		ctOutput, _ := btp.CiphertextMatricesBootstrappingMT(ctOutputLayernorm, btpEval, btpParams, numThreads)
		stop() // 结束计时
		fmt.Println("Bootstrapping for OutputLayernorm finished")

		// 解密输出
		ptDec, _ := he.DecryptCiphertextMatricesMT(ctOutput, modelParams, ckksParams, dec, ecd, numThreads)
		ptDecode := ecdmat.DecodeDense(ptDec, modelParams)
		utils.PrintMat(ptDecode[0])
		filePath := fmt.Sprintf(outputFilePath+"/layer_%d_output.csv", i)
		_ = utils.SaveMatrixToCSV(ptDecode[0], filePath)

		ctX = he.DeepCopyCiphertextMatrices(ctOutput)

		layerElapsed := time.Since(layerStart)
		fmt.Printf("Layer %d Computation Time: %s\n", i, layerElapsed)
	}

	fmt.Printf("================== Pooler  ==================\n")
	// Pooler Output (同理可加时间统计)
	poolerStart := time.Now()

	// Read Pooler Parameters
	poolerDenseWeight, poolerDenseBias, err := utils.ReadPoolerParameters(modelParams)
	if err != nil {
		panic(fmt.Sprintf("failed to read pooler parameters: %v", err))
	}

	// Linear Transformation
	poolerLinearOutput, err := matrix.CiphertextMatricesMultiplyWeightAndAddBiasMT(ctX, poolerDenseWeight.T(), poolerDenseBias, modelParams, ckksParams, eval, numThreads)
	if err != nil {
		panic(err)
	}

	// Tanh Activation
	poolerLinearBeforeTanhOutput, err := tanh.CiphertextMatricesOutputLinear(poolerLinearOutput, modelParams, ckksParams, eval)
	if err != nil {
		panic(err)
	}
	poolerTanhOutput := tanh.CiphertextMatricesTanhChebyshevMT(poolerLinearBeforeTanhOutput, &ckksParams, eval, numThreads)

	// Decrypt and Decode Pooler Output
	ptPoolerOutput, err := he.DecryptCiphertextMatricesMT(poolerTanhOutput, modelParams, ckksParams, dec, ecd, numThreads)
	if err != nil {
		panic(err)
	}
	ptPoolerDecode := ecdmat.DecodeOutputDenseToMatrix(ptPoolerOutput, modelParams)
	fmt.Println("Pooler output:")
	utils.PrintMat(ptPoolerDecode)

	// Save Pooler Output to CSV
	err = utils.SaveMatrixToCSV(ptPoolerDecode, outputFilePath+"/bert_output.csv")
	if err != nil {
		fmt.Println("保存失败:", err)
	}

	poolerElapsed := time.Since(poolerStart)
	fmt.Printf("Pooler Computation Time: %s\n", poolerElapsed)

	fmt.Printf("================== Classifier  ==================\n")
	classifierWeight, classifierBias, err := utils.ReadClassifierParameters(modelParams)
	if err != nil {
		panic(fmt.Sprintf("failed to read classifier parameters: %v", err))
	}

	classifierOut := utils.ClassifierDense(ptPoolerDecode, classifierWeight, classifierBias)
	err = utils.SaveMatrixToCSV(classifierOut, outputFilePath+"/bert_logist.csv")
	if err != nil {
		fmt.Println("保存失败:", err)
	}

	prof.Report(modelParams.NumBatch)
	fmt.Printf("==========   Wall-Clock Total: %s   ==========\n", time.Since(totalStart))
}

func TestCiphertextMatricesMultiplyWeightAndAddBiasMT(paramType string, rowValue int, modelPath string) {
	fmt.Print("----------Task: BERT-Tiny Model Inference Under The CKKS Scheme " + paramType + "----------\n")

	/* Group1: Generate the model parameters, configurations and data */
	ckksParams, btpParams, modelParams, err := configs.InitBertTiny(paramType, rowValue, modelPath)
	if err != nil {
		panic(err)
	}

	numThreads := runtime.NumCPU()
	runtime.GOMAXPROCS(numThreads)
	fmt.Printf("GOMAXPROCS is set to: %d\n", runtime.GOMAXPROCS(0))

	// outputFilePath := "Output/" + modelParams.ModelPath

	mat, err := utils.ReadInput(modelParams, "embedded_inputs.csv")
	if err != nil {
		panic("failed to read input: " + err.Error())
	}
	mat = utils.PadOrTruncateMatrix(mat, modelParams)
	inputMats := utils.MatrixToBatchMats(mat, modelParams)

	// === 密钥生成耗时统计 ===
	keyGenStart := time.Now()
	ecd, enc, eval, dec, btpEval := he.GenerateKeysAndBtsKeysMT(ckksParams, btpParams, modelParams, numThreads)
	_ = dec
	_ = btpEval
	keyGenElapsed := time.Since(keyGenStart)
	fmt.Printf("Key Generation & Bootstrapping Keys Time: %s\n", keyGenElapsed)

	// === 输入加密耗时统计 ===
	encryptStart := time.Now()
	encodedInputs := ecdmat.EncodeDense(inputMats, modelParams)
	ctX, err := he.EncryptInputMatricesMT(encodedInputs, modelParams, ckksParams, enc, ecd, numThreads)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt input matrices: %v", err))
	}
	encryptElapsed := time.Since(encryptStart)
	fmt.Printf("Input Encryption Time: %s\n", encryptElapsed)

	layerParams, err := utils.ReadLayerParameters(modelParams, 0)
	if err != nil {
		panic(fmt.Sprintf("failed to read layer parameters for layer %d: %v", 0, err))
	}

	//=== 矩阵乘法耗时 ===
	matrixMulStart := time.Now()
	ctQ, err := matrix.CiphertextMatricesMultiplyWeightAndAddBias(ctX, layerParams.LayerAttentionSelfKeyWeight.T(), layerParams.LayerAttentionSelfKeyBias, modelParams, ckksParams, eval)
	if err != nil {
		panic(err)
	}
	matrixElapsed := time.Since(matrixMulStart)
	fmt.Printf("CiphertextMatricesMultiplyWeightAndAddBiasMT Time: %s\n", matrixElapsed)

	ctQdec, err := he.DecryptCiphertextMatrices(ctQ, modelParams, ckksParams, dec, ecd)
	if err != nil {
		panic(err)
	}

	ctQdecd := ecdmat.DecodeDense(ctQdec, modelParams)
	utils.PrintMat(ctQdecd[0])
}
