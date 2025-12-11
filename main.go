package main

import (
	"Arion/configs"
	"Arion/pkg/bert"
	"Arion/pkg/btp"
	"Arion/pkg/ecdmat"
	"Arion/pkg/he"
	"Arion/pkg/llama"
	"Arion/pkg/math/activation/gelu"
	"Arion/pkg/math/activation/layernorm"
	"Arion/pkg/math/activation/rmsnorm"
	"Arion/pkg/math/matmul"
	"Arion/pkg/math/matrix"
	"Arion/pkg/utils"
	"bufio"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"gonum.org/v1/gonum/mat"
)

func main() {
	reader := bufio.NewReader(os.Stdin)
	fmt.Println("-----------------MENU--------------------")
	fmt.Println("请选择要运行的任务：")
	fmt.Println("  1. Three Matrices Multiply Test (logN=10)")
	fmt.Println("  2. Ciphertext matrix multiply Plaintext matrix")
	fmt.Println("  3. Bert test function ")
	fmt.Println("  4. Bert Inference with Tiny parameters (logN=6, rows=8, batch=4)")
	fmt.Println("  5. Bert Inference with Short parameters (logN=10, rows=128, batch=4)")
	fmt.Println("  6. Bert Inference with Base parameters (logN=16, rows=128, batch=256)")
	fmt.Println("  7. Bert Tiny test function ")
	fmt.Println("  8. Bert-tiny Inference with Tiny parameters (logN=6, rows=8, batch=4)")
	fmt.Println("  9. Bert-tiny Inference with Short parameters (logN=10, rows=128, batch=4)")
	fmt.Println(" 10. Bert-tiny Inference with Base parameters (logN=16, rows=128, batch=256)")
	fmt.Println(" 11. Bert-tiny Inference with Base parameters using MultiThreading (logN=16, rows=128, batch=256)")
	fmt.Println(" 12. Bert Inference with Base parameters using MultiThreading (logN=16, rows=128, batch=256)")
	// fmt.Println(" 13. Llama 3 Attention Inference")
	fmt.Println("  0. 退出")
	fmt.Println("-----------------------------------------")
	fmt.Print("输入序号: ")
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)
	choice, err := strconv.Atoi(input)
	if err != nil {
		fmt.Println("输入无效，请重新输入。")
		return
	}
	switch choice {
	case 1:
		// 最好单独测试每一个类型
		// runThreeMatricMultiply(13, 32, 64)
		// runThreeMatricMultiply(13, 64, 64)
		// runThreeMatricMultiply(13, 128, 64)
		// runThreeMatricMultiply(13, 256, 64)
		runThreeMatricMultiply(13, 128, 64)
		runThreeMatricMultiply(13, 128, 128)

	case 2:
		bert.TestCiphertextMatricesMultiplyWeightAndAddBiasMT("short", 128, "bert_base_data")
	case 3:
		CiphertextInferenceBertSelfAttention() // 计算结果正确
		// CiphertextInferenceSelfOutput() // 计算结果正确
		// CiphertextInferenceBertIntermediate() // 计算结果正确，但是Gelu函数的degree是255，且在0点附近的精确度不够
		// CiphertextInferenceBertOutput() // 计算结果正确
	case 4:
		bert.InferenceBert("tiny", 8, "bert_base_data")
	case 5:
		bert.InferenceBert("short", 128, "bert_base_data")
	case 6:
		bert.InferenceBert("base", 128, "bert_base_data")
	case 7:
		CiphertextInferenceBertTinySelfAttention() // 计算结果正确
		// CiphertextInferenceBertTinySelfOutput() // 计算结果正确
		// CiphertextInferenceBertTinyIntermediate() // 计算结果正确，但是Gelu函数的degree是255，且在0点附近的精确度不够
		// CiphertextInferenceBertTinyOutput() // 计算结果正确
	case 8:
		bert.InferenceBertTiny("tiny", 8, "bert_tiny_data")
	case 9:
		bert.InferenceBertTiny("short", 128, "bert_tiny_data_rte")
	case 10:
		bert.InferenceBertTiny("base", 128, "bert_tiny_data")
	case 11:
		bert.InferenceBertTinyMT("base", 128, "bert_tiny_data")
		bert.InferenceBertTinyClassifierMT("base", 128, "bert_tiny_data_qnli")
		bert.InferenceBertTinyClassifierMT("base", 128, "bert_tiny_data_rte")
		bert.InferenceBertTinyClassifierMT("base", 128, "bert_tiny_data_sst2")
		// bert.InferenceBertTinyMT("short", 128, "bert_tiny_data")
		// bert.InferenceBertTinyClassifierMT("short", 128, "bert_tiny_data_qnli")
		// bert.InferenceBertTinyClassifierMT("short", 128, "bert_tiny_data_rte")
		// bert.InferenceBertTinyClassifierMT("short", 128, "bert_tiny_data_sst2")
	case 12:
		bert.InferenceBertMT("base", 128, "bert_base_data_5")
		// bert.InferenceBertMT("short", 128, "bert_base_data_5")
	case 13:
		// CiphertextInferenceLlamaAttention("tiny", 8, "llama3_8b")
		// CiphertextInferenceLlamaTest("tiny", 8, "llama3_8b")
	case 0:
		fmt.Println("程序已退出。")
		return
	default:
		fmt.Println("输入无效，请重新输入。")
	}
	fmt.Println("\n任务结束。")
}

func CiphertextInferenceBertSelfAttention() {
	// This function is a placeholder for the Bert model ciphertext inference logic.
	// It should be implemented with the actual logic for performing inference on the Bert model.
	fmt.Println("----------Task: BERT Model Inference Under The CKKS Scheme Attention----------")

	// Step 1. Generate the model parameters and configurations
	ckksParams, btpParams, modelParams, err := configs.InitBert("short", 128, "bert_base_data")
	if err != nil {
		panic(err)
	}

	// Step 2.1 Read the input data from the CSV file
	mat, err := utils.ReadInput(modelParams, "embedded_inputs.csv")
	if err != nil {
		panic(err)
	}
	row, col := mat.Dims()
	fmt.Printf("Input matrix dimensions: %d rows, %d columns\n", row, col)
	// Pad or truncate the matrix to match the model parameters
	mat = utils.PadOrTruncateMatrix(mat, modelParams)
	row, col = mat.Dims()
	fmt.Printf("Pad matrix dimensions: %d rows, %d columns\n", row, col)
	// Generate the input matrices for Bert model inference
	inputMatrices := utils.MatrixToBatchMats(mat, modelParams)

	// Step 2.2 Read the layer parameters for the first layer
	layerParams, err := utils.ReadLayerParameters(modelParams, 2)
	if err != nil {
		panic(fmt.Sprintf("failed to read layer parameters: %v", err))
	}
	utils.PrintLayerParametersDims(layerParams)

	// Step 3.1 Encode the input matrices
	encodedInputs := ecdmat.EncodeDense(inputMatrices, modelParams)
	// utils.PrintFirstOfVectors(encodedInputs, 10)
	decodedInputs := ecdmat.DecodeDense(encodedInputs, modelParams)
	utils.PrintMat(decodedInputs[0])

	// Step 3.2 Generate the keys for the CKKS scheme
	ecd, enc, eval, dec, btpEval := he.GenerateKeysAndBtsKeys(ckksParams, btpParams, modelParams)
	_ = dec
	_ = btpEval
	// Step 3.3 Encrypt the input matrices
	ctX, err := he.EncryptInputMatrices(encodedInputs, modelParams, ckksParams, enc, ecd)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt input matrices: %v", err))
	}

	// Step 4. 计算Q、K、V
	ctQ, ctK, ctV, err := bert.ComputeQKV(ctX, layerParams, modelParams, ckksParams, eval)
	if err != nil {
		panic(fmt.Sprintf("failed to compute QKV: %v", err))
	}
	_ = ctV
	_ = ctQ

	ctQsplit := matrix.SplitCiphertextMatricesByHeads(ctQ, modelParams)
	ctKsplit := matrix.SplitCiphertextMatricesByHeads(ctK, modelParams)
	ctVsplit := matrix.SplitCiphertextMatricesByHeads(ctV, modelParams)
	ctAttentionHeads := make([]*he.CiphertextMatrices, modelParams.NumHeads)
	for j := 0; j < modelParams.NumHeads; j++ {
		ctAttention, ctAdd, err := bert.ComputeAttention(ctQsplit[j], ctKsplit[j], ctVsplit[j], modelParams, &ckksParams, ecd, enc, eval, btpEval)
		if err != nil {
			panic(fmt.Sprintf("failed to compute attention for head %d in layer %d: %v", j, 0, err))
		}
		_ = ctAdd // ctAdd is not used, but computed for consistency
		ctAttentionHeads[j] = ctAttention
		fmt.Printf("Layer %d, Head %d: Computed attention with %d ciphertexts\n", 0, j, len(ctAttention.Ciphertexts))
	}
	// 3.3) Combine the attention heads
	ctAttention := matrix.MergeCiphertextMatricesByHeads(ctAttentionHeads)
	fmt.Println("Attention level:", ctAttention.Ciphertexts[0].Level())

	ctQKTdec, err := he.DecryptCiphertextMatrices(ctAttention, modelParams, ckksParams, dec, ecd)
	if err != nil {
		panic(err)
	}

	ctQKTdecd := ecdmat.DecodeDense(ctQKTdec, modelParams)
	utils.PrintMat(ctQKTdecd[0])

	fmt.Println("QK attention computation finished.")
}

func CiphertextInferenceSelfOutput() {
	// This function is a placeholder for the Bert model ciphertext inference logic.
	// It should be implemented with the actual logic for performing inference on the Bert model.
	fmt.Println("----------Task: BERT Model Inference Under The CKKS Scheme SelfOutput----------")

	// Step 1. Generate the model parameters and configurations
	ckksParams, _, modelParams, err := configs.InitBert("short", 128, "bert_base_data")
	if err != nil {
		panic(err)
	}

	// Step 2.1 Read the input data from the CSV file
	matInput, err := utils.ReadInput(modelParams, "embedded_inputs.csv")
	if err != nil {
		panic(err)
	}
	row, col := matInput.Dims()
	fmt.Printf("Input matrix dimensions: %d rows, %d columns\n", row, col)
	// Pad or truncate the matrix to match the model parameters
	matInput = utils.PadOrTruncateMatrix(matInput, modelParams)
	row, col = matInput.Dims()
	fmt.Printf("Pad matrix dimensions: %d rows, %d columns\n", row, col)
	// Generate the input matrices for Bert model inference
	inputMatrices := utils.MatrixToBatchMats(matInput, modelParams)

	// Step 2.1 Read the input data from the CSV file
	selfInputPath := filepath.Join(modelParams.ModelPath, "layer_0", "Attention", "SelfOutput", "allresults", "self_output_inputs.csv")
	mat, err := utils.ReadCSVToMatrix(selfInputPath)
	if err != nil {
		panic(err)
	}
	row, col = mat.Dims()
	fmt.Printf("Input matrix dimensions: %d rows, %d columns\n", row, col)
	// Pad or truncate the matrix to match the model parameters
	mat = utils.PadOrTruncateMatrix(mat, modelParams)
	row, col = mat.Dims()
	fmt.Printf("Pad matrix dimensions: %d rows, %d columns\n", row, col)
	// Generate the input matrices for Bert model inference
	selfOutputInputMatrices := utils.MatrixToBatchMats(mat, modelParams)

	// Step 2.2 Read the layer parameters for the first layer
	layerParams, err := utils.ReadLayerParameters(modelParams, 0)
	if err != nil {
		panic(fmt.Sprintf("failed to read layer parameters: %v", err))
	}
	utils.PrintLayerParametersDims(layerParams)

	// Step 3.1 Encode the input matrices
	encodedInputs := ecdmat.EncodeDense(inputMatrices, modelParams)
	encodedSelfOutputInputs := ecdmat.EncodeDense(selfOutputInputMatrices, modelParams)
	// utils.PrintFirstOfVectors(encodedInputs, 10)
	decodedInputs := ecdmat.DecodeDense(encodedSelfOutputInputs, modelParams)
	fmt.Println("Decoded input matrices:")
	utils.PrintMat(decodedInputs[0])

	// Step 3.2 Generate the keys for the CKKS scheme
	ecd, enc, eval, dec := he.GenerateKeys(ckksParams, modelParams)
	_ = dec
	// Step 3.3 Encrypt the input matrices
	ctX, err := he.EncryptInputMatrices(encodedSelfOutputInputs, modelParams, ckksParams, enc, ecd)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt input matrices: %v", err))
	}

	ctInput, err := he.EncryptInputMatrices(encodedInputs, modelParams, ckksParams, enc, ecd)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt input matrices: %v", err))
	}

	fmt.Println("Input matrices level:", ctInput.Ciphertexts[0].Level())

	// Step 4. 计算dense
	ctDense, err := matrix.CiphertextMatricesMultiplyWeightAndAddBias(ctX, layerParams.LayerAttentionOutputDenseWeight.T(), layerParams.LayerAttentionOutputDenseBias, modelParams, ckksParams, eval)
	if err != nil {
		panic(err)
	}

	// step 5. residual connection
	ctResidual, err := matrix.CiphertextMatricesAddCiphertextMatrices(ctDense, ctInput, eval)
	if err != nil {
		panic(fmt.Sprintf("failed to add ciphertext matrices: %v", err))
	}

	// fmt.Println("Already residual connection")
	// Step 6. 计算layer norm
	ctLayernorm, err := layernorm.LayerNormSelfAttentionOutput(ctResidual, layerParams.LayerAttentionOutputLayerNormWeight, layerParams.LayerAttentionOutputLayerNormBias, modelParams, ckksParams, ecd, eval)
	if err != nil {
		panic(fmt.Sprintf("failed to compute layer norm: %v", err))
	}

	fmt.Println("ctLayernorm level:", ctLayernorm.Ciphertexts[0].Level())

	ctQKTdec, err := he.DecryptCiphertextMatrices(ctLayernorm, modelParams, ckksParams, dec, ecd)
	if err != nil {
		panic(err)
	}

	ctQKTdecd := ecdmat.DecodeDense(ctQKTdec, modelParams)
	utils.PrintMat(ctQKTdecd[0])

	fmt.Println("SelfOutput computation finished.")
}

func CiphertextInferenceBertIntermediate() {
	// This function is a placeholder for the Bert model ciphertext inference logic.
	// It should be implemented with the actual logic for performing inference on the Bert model.
	fmt.Println("----------Task: BERT Model Inference Under The CKKS Scheme Intermediate----------")

	// Step 1. Generate the model parameters and configurations
	ckksParams, _, modelParams, err := configs.InitBert("tiny", 8, "bert_base_data")
	if err != nil {
		panic(err)
	}

	// Step 2.1 Read the input data from the CSV file
	selfInputPath := filepath.Join(modelParams.ModelPath, "layer_0", "Intermediate", "allresults", "intermediate_inputs.csv")
	mat, err := utils.ReadCSVToMatrix(selfInputPath)
	if err != nil {
		panic(err)
	}
	row, col := mat.Dims()
	fmt.Printf("Input matrix dimensions: %d rows, %d columns\n", row, col)
	// Pad or truncate the matrix to match the model parameters
	mat = utils.PadOrTruncateMatrix(mat, modelParams)
	row, col = mat.Dims()
	fmt.Printf("Pad matrix dimensions: %d rows, %d columns\n", row, col)
	// Generate the input matrices for Bert model inference
	inputMatrices := utils.MatrixToBatchMats(mat, modelParams)

	// Step 2.2 Read the layer parameters for the first layer
	layerParams, err := utils.ReadLayerParameters(modelParams, 0)
	if err != nil {
		panic(fmt.Sprintf("failed to read layer parameters: %v", err))
	}
	utils.PrintLayerParametersDims(layerParams)

	// Step 3.1 Encode the input matrices
	encodedInputs := ecdmat.EncodeDense(inputMatrices, modelParams)
	// utils.PrintFirstOfVectors(encodedInputs, 10)
	decodedInputs := ecdmat.DecodeDense(encodedInputs, modelParams)
	utils.PrintMat(decodedInputs[0])

	// Step 3.2 Generate the keys for the CKKS scheme
	ecd, enc, eval, dec := he.GenerateKeys(ckksParams, modelParams)
	_ = dec
	// Step 3.3 Encrypt the input matrices
	ctX, err := he.EncryptInputMatrices(encodedInputs, modelParams, ckksParams, enc, ecd)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt input matrices: %v", err))
	}

	// Intermediate
	ctIntermediate, err := matrix.CiphertextMatricesMultiplyWeightAndAddBias(ctX, layerParams.LayerIntermediateDenseWeight.T(), layerParams.LayerIntermediateDenseBias, modelParams, ckksParams, eval)
	if err != nil {
		panic(err)
	}

	// GeLU
	ctGeLU := gelu.CiphertextMatricesGeluChebyshev(ctIntermediate, &ckksParams, eval, modelParams.GeluMinValue, modelParams.GeluMaxValue, modelParams.GeluDegree)

	// decryption
	ptDec, err := he.DecryptCiphertextMatrices(ctGeLU, modelParams, ckksParams, dec, ecd)
	if err != nil {
		panic(err)
	}

	ctDec := ecdmat.DecodeDense(ptDec, modelParams)
	utils.PrintMat(ctDec[0])

	fmt.Println("Intermediate computation finished.")
}

func CiphertextInferenceBertOutput() {
	// This function is a placeholder for the Bert model ciphertext inference logic.
	// It should be implemented with the actual logic for performing inference on the Bert model.
	fmt.Println("----------Task: BERT Model Inference Under The CKKS Scheme Output----------")

	// Step 1. Generate the model parameters and configurations
	ckksParams, _, modelParams, err := configs.InitBert("tiny", 8, "bert_base_data")
	if err != nil {
		panic(err)
	}

	var layer = 9
	// Step 2.1 Read the input data from the CSV file
	SelfLayernormPath := filepath.Join(modelParams.ModelPath, "layer_"+strconv.Itoa(layer), "Attention", "SelfOutput", "allresults", "real_self_output.csv")
	matSelfLayernorm, err := utils.ReadCSVToMatrix(SelfLayernormPath)
	if err != nil {
		panic(err)
	}
	row, col := matSelfLayernorm.Dims()
	fmt.Printf("Input matrix dimensions: %d rows, %d columns\n", row, col)
	// Pad or truncate the matrix to match the model parameters
	matSelfLayernorm = utils.PadOrTruncateMatrix(matSelfLayernorm, modelParams)
	row, col = matSelfLayernorm.Dims()
	fmt.Printf("Pad matrix dimensions: %d rows, %d columns\n", row, col)
	// Generate the input matrices for Bert model inference
	selfLayernormMatrices := utils.MatrixToBatchMats(matSelfLayernorm, modelParams)

	// Step 2.1 Read the input data from the CSV file
	outputInputPath := filepath.Join(modelParams.ModelPath, "layer_"+strconv.Itoa(layer), "Output", "allresults", "final_output_inputs.csv")
	mat, err := utils.ReadCSVToMatrix(outputInputPath)
	if err != nil {
		panic(err)
	}
	row, col = mat.Dims()
	fmt.Printf("Input matrix dimensions: %d rows, %d columns\n", row, col)
	// Pad or truncate the matrix to match the model parameters
	mat = utils.PadOrTruncateMatrix(mat, modelParams)
	row, col = mat.Dims()
	fmt.Printf("Pad matrix dimensions: %d rows, %d columns\n", row, col)
	// Generate the input matrices for Bert model inference
	outputInputMatrices := utils.MatrixToBatchMats(mat, modelParams)

	// Step 2.2 Read the layer parameters for the first layer
	layerParams, err := utils.ReadLayerParameters(modelParams, layer)
	if err != nil {
		panic(fmt.Sprintf("failed to read layer parameters: %v", err))
	}
	utils.PrintLayerParametersDims(layerParams)

	// Step 3.1 Encode the input matrices
	encodedSelfLayernormMatrices := ecdmat.EncodeDense(selfLayernormMatrices, modelParams)
	encodedOutputInputs := ecdmat.EncodeDense(outputInputMatrices, modelParams)
	// utils.PrintFirstOfVectors(encodedInputs, 10)
	decodedInputs := ecdmat.DecodeDense(encodedOutputInputs, modelParams)
	fmt.Println("Decoded input matrices:")
	utils.PrintMat(decodedInputs[0])

	// Step 3.2 Generate the keys for the CKKS scheme
	ecd, enc, eval, dec := he.GenerateKeys(ckksParams, modelParams)
	_ = dec
	// Step 3.3 Encrypt the input matrices
	ctX, err := he.EncryptInputMatrices(encodedOutputInputs, modelParams, ckksParams, enc, ecd)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt input matrices: %v", err))
	}

	ctSelfOutputLayernorm, err := he.EncryptInputMatrices(encodedSelfLayernormMatrices, modelParams, ckksParams, enc, ecd)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt input matrices: %v", err))
	}

	// Step 4. 计算dense
	ctOutputDense, err := matrix.CiphertextMatricesMultiplyWeightAndAddBias(ctX, layerParams.LayerOutputDenseWeight.T(), layerParams.LayerOutputDenseBias, modelParams, ckksParams, eval)
	if err != nil {
		panic(err)
	}

	ptDenseDec, err := he.DecryptCiphertextMatrices(ctOutputDense, modelParams, ckksParams, dec, ecd)
	if err != nil {
		panic(err)
	}

	ptDenseDecode := ecdmat.DecodeDense(ptDenseDec, modelParams)
	fmt.Println("MatMul Dense result:")
	utils.PrintMat(ptDenseDecode[0])

	// step 5. residual connection
	ctResidual, err := matrix.CiphertextMatricesAddCiphertextMatrices(ctOutputDense, ctSelfOutputLayernorm, eval)
	if err != nil {
		panic(fmt.Sprintf("failed to add ciphertext matrices: %v", err))
	}

	ptResidualDec, err := he.DecryptCiphertextMatrices(ctResidual, modelParams, ckksParams, dec, ecd)
	if err != nil {
		panic(err)
	}

	ptResidualDecode := ecdmat.DecodeDense(ptResidualDec, modelParams)
	fmt.Println("Residual connection result:")
	utils.PrintMat(ptResidualDecode[0])

	// Step 6. 计算layer norm
	ctLayernorm, err := layernorm.LayerNormOutputTest(ctResidual, layerParams.LayerOutputLayerNormWeight, layerParams.LayerOutputLayerNormBias, modelParams, ckksParams, ecd, eval, dec)
	if err != nil {
		panic(fmt.Sprintf("failed to compute layer norm: %v", err))
	}

	ptDec, err := he.DecryptCiphertextMatrices(ctLayernorm, modelParams, ckksParams, dec, ecd)
	if err != nil {
		panic(err)
	}

	ptDecode := ecdmat.DecodeDense(ptDec, modelParams)
	fmt.Println("layer norm result:")
	utils.PrintMat(ptDecode[0])

	fmt.Println("Output computation finished.")
}

func runAttentionWithApproxMax(modelParams *configs.ModelParams, logN int) {
	fmt.Println("===================================")
	fmt.Printf(" * 正在运行 ParamGroup LogN=%d ...\n", logN)
	fmt.Println("===================================")
	benchmarkAttentionWithApproxMax(modelParams, logN)
}

func benchmarkAttentionWithApproxMax(modelParams *configs.ModelParams, logN int) {

	// 初始化CKKS参数
	ckksParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            logN,
		LogQ:            []int{58, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
		LogP:            []int{60, 60, 60},
		LogDefaultScale: 46,
	})
	if err != nil {
		fmt.Printf("CKKS参数初始化失败: %v\n", err)
	}

	// 随机生成输入矩阵
	inputQ := mat.NewDense(modelParams.NumRealRow, modelParams.NumCol, nil)
	inputK := mat.NewDense(modelParams.NumRealRow, modelParams.NumCol, nil)
	inputV := mat.NewDense(modelParams.NumRealRow, modelParams.NumCol, nil)
	for i := 0; i < modelParams.NumRealRow; i++ {
		for j := 0; j < modelParams.NumCol; j++ {
			inputQ.Set(i, j, rand.Float64()*2-1)
			inputK.Set(i, j, rand.Float64()*2-1)
			inputV.Set(i, j, rand.Float64()*2-1)
		}
	}

	// 编码、批处理
	inputQ = utils.PadOrTruncateMatrix(inputQ, modelParams)
	inputK = utils.PadOrTruncateMatrix(inputK, modelParams)
	inputV = utils.PadOrTruncateMatrix(inputV, modelParams)
	inputMatsQ := utils.MatrixToBatchMats(inputQ, modelParams)
	inputMatsK := utils.MatrixToBatchMats(inputK, modelParams)
	inputMatsV := utils.MatrixToBatchMats(inputV, modelParams)
	encodedInputsQ := ecdmat.EncodeDense(inputMatsQ, modelParams)
	encodedInputsK := ecdmat.EncodeDense(inputMatsK, modelParams)
	encodedInputsV := ecdmat.EncodeDense(inputMatsV, modelParams)

	// 内存统计：开始
	var mEnd runtime.MemStats

	// 统计生成密钥时间
	startKey := time.Now()
	ecd, enc, eval, dec := he.GenerateKeys(ckksParams, modelParams)
	keyGenTime := time.Since(startKey)
	fmt.Printf("CKKS Paramter: LogN: %d - LogQP %.2f - LogSlots: %d\n", ckksParams.LogN(), ckksParams.LogQP(), ckksParams.LogMaxSlots())

	// 加密
	ctQ, err := he.EncryptInputMatrices(encodedInputsQ, modelParams, ckksParams, enc, ecd)
	if err != nil {
		fmt.Printf("加密Q失败: %v\n", err)
	}
	ctK, err := he.EncryptInputMatrices(encodedInputsK, modelParams, ckksParams, enc, ecd)
	if err != nil {
		fmt.Printf("加密K失败: %v\n", err)
	}
	ctV, err := he.EncryptInputMatrices(encodedInputsV, modelParams, ckksParams, enc, ecd)
	if err != nil {
		fmt.Printf("加密V失败: %v\n", err)
	}

	fmt.Println("初始层数：", ctQ.Ciphertexts[0].Level())

	// 统计Attention计算时间
	startAtt := time.Now()
	ctExpQKTMulV, _, err := matrix.CiphertextMatricesComputeAttentionWithBSGSAndApproxMax(ctQ, ctK, ctV, modelParams, ckksParams, eval)
	if err != nil {
		fmt.Printf("密文矩阵attention计算失败: %v\n", err)
	}
	attTime := time.Since(startAtt)
	fmt.Println("完成计算层数：", ctExpQKTMulV.Ciphertexts[0].Level())

	// 解密
	decY, err := he.DecryptCiphertextMatrices(ctExpQKTMulV, modelParams, ckksParams, dec, ecd)
	if err != nil {
		fmt.Printf("解密失败: %v\n", err)
	}

	// 解码
	matsY := ecdmat.DecodeDense(decY, modelParams)
	// fmt.Println("cipherInferenceAtten: ")
	// utils.PrintMat(matsY[0])

	// 内存统计：结束
	runtime.ReadMemStats(&mEnd)

	// 明文 attention: exp(QK^T - approxMax) × V
	// 明文直接计算 Q × K^T
	expectedQK := mat.NewDense(modelParams.NumRow, modelParams.NumRow, nil)
	kT := mat.DenseCopyOf(inputK.T())
	expectedQK.Product(inputQ, kT)

	// 明文近似最大值计算
	rowMeans := make([]float64, modelParams.NumRow)
	rowVars := make([]float64, modelParams.NumRow)
	rowStds := make([]float64, modelParams.NumRow)
	approxMax := make([]float64, modelParams.NumRow)
	realRow := float64(modelParams.NumRealRow)
	// 明文近似最大值计算（只统计有效token）
	for i := 0; i < modelParams.NumRow; i++ {
		sum := 0.0
		for j := 0; j < int(modelParams.NumRealRow); j++ {
			sum += expectedQK.At(i, j)
		}
		rowMeans[i] = sum / realRow
		varSum := 0.0
		for j := 0; j < int(modelParams.NumRealRow); j++ {
			varSum += math.Pow(expectedQK.At(i, j)-rowMeans[i], 2)
		}
		rowVars[i] = varSum / realRow
		rowStds[i] = 1.25 + 0.1*rowVars[i]
		constValue := 2.08
		approxMax[i] = rowMeans[i] + rowStds[i]*constValue
		// fmt.Println(rowMeans[i], "  ", rowVars[i], "  ", rowStds[i], "  ", varSum)
	}
	// t.Logf("明文 attention 近似最大值: %v", approxMax)

	// exp(QK^T - approxMax)，超出部分直接设为0
	expMat := mat.NewDense(modelParams.NumRow, modelParams.NumRow, nil)
	for i := 0; i < modelParams.NumRow; i++ {
		for j := 0; j < modelParams.NumRow; j++ {
			if i < int(modelParams.NumRealRow) && j < int(modelParams.NumRealRow) {
				val := math.Exp(expectedQK.At(i, j) - approxMax[i])
				expMat.Set(i, j, val)
			} else {
				expMat.Set(i, j, 0) // mask掉无效token（行或列超出都设为0）
			}
		}
	}

	// exp(QK^T - approxMax)，超出部分直接设为0
	expectedAtt := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
	expectedAtt.Product(expMat, inputV)
	// fmt.Println("expectedAtten: ")
	// utils.PrintMat(expectedAtt)

	// 统计最大误差
	var maxErr float64
	got := matsY[0]
	row, col := got.Dims()
	for i := 0; i < row; i++ {
		for j := 0; j < col; j++ {
			err := math.Abs(got.At(i, j) - expectedAtt.At(i, j))
			if err > maxErr {
				maxErr = err
			}
		}
	}

	// ... 输出统计信息 ...
	fmt.Println("================= 统计信息 =================")
	fmt.Printf("* Batch数量: %d, BSGS大小: (%d,%d)\n", modelParams.NumBatch, modelParams.BabyStep, modelParams.GiantStep)
	qRows, qCols := inputQ.Dims()
	kRows, kCols := inputK.Dims()
	vRows, vCols := inputV.Dims()
	fmt.Printf("* Q矩阵 shape: (%d, %d), ", qRows, qCols)
	fmt.Printf("K矩阵 shape: (%d, %d), ", kRows, kCols)
	fmt.Printf("V矩阵 shape: (%d, %d)\n", vRows, vCols)
	if len(matsY) > 0 {
		r, c := matsY[0].Dims()
		fmt.Printf("* exp(QK^T - approxMax)× V输出 shape: (%d, %d)\n", r, c)
	}
	fmt.Printf("* 密钥生成耗时: %v\n", keyGenTime)
	fmt.Printf("* exp(QK^T - approxMax)× V计算总时间: %v\n", attTime)
	fmt.Printf("* exp(QK^T - approxMax)× V计算均摊时间: %v\n", attTime/time.Duration(modelParams.NumBatch))
	peakMemGB := float64(mEnd.Sys) / 1024.0 / 1024.0 / 1024.0
	totalMemGB := float64(mEnd.Alloc) / 1024.0 / 1024.0 / 1024.0
	fmt.Printf("* 峰值内存占用: %.2f GB\n", peakMemGB)
	fmt.Printf("* 总内存占用: %.2f GB\n", totalMemGB)
	fmt.Printf("* 明文与密文(decode) exp(QK^T - approxMax)× V最大绝对误差: %.6e\n", maxErr)
	fmt.Println("===============================================")
}

func runThreeMatricMultiply(LogN, numRow, numCol int) {
	// 参数设置
	babyStep, gaintStep := configs.ChooseSteps(numRow)

	// 初始化CKKS参数
	ckksParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            LogN,              // ring degree = 2^16, log QP <= 1748
		LogQ:            []int{50, 40, 40}, // moduli chain
		LogP:            []int{60},         // aux moduli for bootstrapping
		LogDefaultScale: 40,
		Xs:              ring.Ternary{H: 192}, // secret key distribution
	})
	if err != nil {
		panic(err)
	}

	maxSlots := ckksParams.MaxSlots() // = 2^(logN-1)
	numBatch := int(maxSlots / numRow)

	modelParams := &configs.ModelParams{
		NumBatch:  numBatch,
		NumRow:    numRow,
		NumCol:    numCol,
		BabyStep:  babyStep,
		GiantStep: gaintStep,
	}

	ecd, enc, eval, dec := he.GenerateKeys(ckksParams, modelParams)

	matmul.RunThreeMatricMultiplyNormalMT(modelParams, ckksParams, ecd, enc, eval, dec)
	matmul.RunThreeMatricMultiplyBSGSMT(modelParams, ckksParams, ecd, enc, eval, dec)
}

func runThreeMatricMultiplyMT(LogN, numRow, numCol int) {
	// 参数设置
	babyStep, gaintStep := configs.ChooseSteps(numRow)

	ckksParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            LogN,                                                              // ring degree = 2^16, log QP <= 1748
		LogQ:            []int{58, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45}, // moduli chain
		LogP:            []int{60, 60, 60, 60},                                             // aux moduli for bootstrapping
		LogDefaultScale: 45,
		Xs:              ring.Ternary{H: 192}, // secret key distribution
	})
	if err != nil {
		panic(err)
	}

	maxSlots := ckksParams.MaxSlots() // = 2^(logN-1)
	numBatch := int(maxSlots / numRow)

	modelParams := &configs.ModelParams{
		NumBatch:  numBatch,
		NumRow:    numRow,
		NumCol:    numCol,
		BabyStep:  babyStep,
		GiantStep: gaintStep,
	}

	ecd, enc, eval, dec := he.GenerateKeys(ckksParams, modelParams)

	matmul.RunThreeMatricMultiplyNormalMT(modelParams, ckksParams, ecd, enc, eval, dec)
	matmul.RunThreeMatricMultiplyBSGSMT(modelParams, ckksParams, ecd, enc, eval, dec)
}

func CiphertextInferenceBertTinySelfAttention() {
	// This function is a placeholder for the Bert model ciphertext inference logic.
	// It should be implemented with the actual logic for performing inference on the Bert model.
	fmt.Println("----------Task: BERT-Tiny Model Inference Under The CKKS Scheme Attention----------")

	// Step 1. Generate the model parameters and configurations
	ckksParams, btpParams, modelParams, err := configs.InitBertTiny("short", 128, "bert_tiny_data_rte")
	if err != nil {
		panic(err)
	}

	// Step 2.1 Read the input data from the CSV file
	mat, err := utils.ReadInput(modelParams, "embedded_inputs.csv")
	if err != nil {
		panic(err)
	}
	row, col := mat.Dims()
	fmt.Printf("Input matrix dimensions: %d rows, %d columns\n", row, col)
	// Pad or truncate the matrix to match the model parameters
	mat = utils.PadOrTruncateMatrix(mat, modelParams)
	row, col = mat.Dims()
	fmt.Printf("Pad matrix dimensions: %d rows, %d columns\n", row, col)
	// Generate the input matrices for Bert model inference
	inputMatrices := utils.MatrixToBatchMats(mat, modelParams)

	// Step 2.2 Read the layer parameters for the first layer
	layerParams, err := utils.ReadLayerParameters(modelParams, 1)
	if err != nil {
		panic(fmt.Sprintf("failed to read layer parameters: %v", err))
	}
	utils.PrintLayerParametersDims(layerParams)

	// Step 3.1 Encode the input matrices
	encodedInputs := ecdmat.EncodeDense(inputMatrices, modelParams)
	// utils.PrintFirstOfVectors(encodedInputs, 10)
	decodedInputs := ecdmat.DecodeDense(encodedInputs, modelParams)
	utils.PrintMat(decodedInputs[0])
	// fmt.Println(modelParams.NumRealRow)
	// fmt.Println(modelParams.NumBatch)

	// Step 3.2 Generate the keys for the CKKS scheme
	ecd, enc, eval, dec, btpEval := he.GenerateKeysAndBtsKeys(ckksParams, btpParams, modelParams)
	_ = dec
	_ = btpEval
	// Step 3.3 Encrypt the input matrices
	ctX, err := he.EncryptInputMatrices(encodedInputs, modelParams, ckksParams, enc, ecd)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt input matrices: %v", err))
	}

	// Step 4. 计算Q、K、V
	// fmt.Println("LayerAttentionSelfKeyWeight:")
	// utils.PrintMat(layerParams.LayerAttentionSelfKeyWeight)
	// fmt.Println("LayerAttentionSelfKeyBias:")
	// utils.PrintMat(layerParams.LayerAttentionSelfKeyBias)
	ctQ, ctK, ctV, err := bert.ComputeQKV(ctX, layerParams, modelParams, ckksParams, eval)
	if err != nil {
		panic(fmt.Sprintf("failed to compute QKV: %v", err))
	}
	_ = ctV
	_ = ctQ

	ctQsplit := matrix.SplitCiphertextMatricesByHeads(ctQ, modelParams)
	ctKsplit := matrix.SplitCiphertextMatricesByHeads(ctK, modelParams)
	ctVsplit := matrix.SplitCiphertextMatricesByHeads(ctV, modelParams)
	ctAttentionHeads := make([]*he.CiphertextMatrices, modelParams.NumHeads)
	for j := 0; j < modelParams.NumHeads; j++ {
		ctAttention, ctAdd, err := bert.ComputeAttention(ctQsplit[j], ctKsplit[j], ctVsplit[j], modelParams, &ckksParams, ecd, enc, eval, btpEval)
		if err != nil {
			panic(fmt.Sprintf("failed to compute attention for head %d in layer %d: %v", j, 0, err))
		}
		ptAdd := he.DecryptCiphertext(ctAdd, ckksParams, dec, ecd)
		fmt.Println("ptadd:", ptAdd[0:10])
		ctAttentionHeads[j] = ctAttention
		fmt.Printf("Layer %d, Head %d: Computed attention with %d ciphertexts\n", 0, j, len(ctAttention.Ciphertexts))
	}
	// 3.3) Combine the attention heads
	ctAttention := matrix.MergeCiphertextMatricesByHeads(ctAttentionHeads)
	fmt.Println("Attention level:", ctAttention.Ciphertexts[0].Level())

	ctQKTdec, err := he.DecryptCiphertextMatrices(ctAttention, modelParams, ckksParams, dec, ecd)
	if err != nil {
		panic(err)
	}

	ctQKTdecd := ecdmat.DecodeDense(ctQKTdec, modelParams)
	utils.PrintMat(ctQKTdecd[0])

	fmt.Println("QK attention computation finished.")
}

func CiphertextInferenceBertTinySelfOutput() {
	// This function is a placeholder for the Bert model ciphertext inference logic.
	// It should be implemented with the actual logic for performing inference on the Bert model.
	fmt.Println("----------Task: BERT-Tiny Model Inference Under The CKKS Scheme SelfOutput----------")

	// Step 1. Generate the model parameters and configurations
	ckksParams, _, modelParams, err := configs.InitBertTiny("short", 128, "bert_tiny_data_sst2")
	if err != nil {
		panic(err)
	}

	// Step 2.1 Read the input data from the CSV file
	matInput, err := utils.ReadInput(modelParams, "embedded_inputs.csv")
	if err != nil {
		panic(err)
	}
	row, col := matInput.Dims()
	fmt.Printf("Input matrix dimensions: %d rows, %d columns\n", row, col)
	// Pad or truncate the matrix to match the model parameters
	matInput = utils.PadOrTruncateMatrix(matInput, modelParams)
	row, col = matInput.Dims()
	fmt.Printf("Pad matrix dimensions: %d rows, %d columns\n", row, col)
	// Generate the input matrices for Bert model inference
	inputMatrices := utils.MatrixToBatchMats(matInput, modelParams)

	// Step 2.1 Read the input data from the CSV file
	selfInputPath := filepath.Join(modelParams.ModelPath, "layer_1", "Attention", "SelfOutput", "allresults", "self_output_inputs.csv")
	mat, err := utils.ReadCSVToMatrix(selfInputPath)
	if err != nil {
		panic(err)
	}
	row, col = mat.Dims()
	fmt.Printf("Input matrix dimensions: %d rows, %d columns\n", row, col)
	// Pad or truncate the matrix to match the model parameters
	mat = utils.PadOrTruncateMatrix(mat, modelParams)
	row, col = mat.Dims()
	fmt.Printf("Pad matrix dimensions: %d rows, %d columns\n", row, col)
	// Generate the input matrices for Bert model inference
	selfOutputInputMatrices := utils.MatrixToBatchMats(mat, modelParams)

	// Step 2.2 Read the layer parameters for the first layer
	layerParams, err := utils.ReadLayerParameters(modelParams, 1)
	if err != nil {
		panic(fmt.Sprintf("failed to read layer parameters: %v", err))
	}
	utils.PrintLayerParametersDims(layerParams)

	// Step 3.1 Encode the input matrices
	encodedInputs := ecdmat.EncodeDense(inputMatrices, modelParams)
	encodedSelfOutputInputs := ecdmat.EncodeDense(selfOutputInputMatrices, modelParams)
	// utils.PrintFirstOfVectors(encodedInputs, 10)
	decodedInputs := ecdmat.DecodeDense(encodedSelfOutputInputs, modelParams)
	fmt.Println("Decoded input matrices:")
	utils.PrintMat(decodedInputs[0])

	// Step 3.2 Generate the keys for the CKKS scheme
	ecd, enc, eval, dec := he.GenerateKeys(ckksParams, modelParams)
	_ = dec
	// Step 3.3 Encrypt the input matrices
	ctX, err := he.EncryptInputMatrices(encodedSelfOutputInputs, modelParams, ckksParams, enc, ecd)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt input matrices: %v", err))
	}

	ctInput, err := he.EncryptInputMatrices(encodedInputs, modelParams, ckksParams, enc, ecd)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt input matrices: %v", err))
	}

	fmt.Println("Input matrices level:", ctInput.Ciphertexts[0].Level())

	// Step 4. 计算dense
	ctDense, err := matrix.CiphertextMatricesMultiplyWeightAndAddBias(ctX, layerParams.LayerAttentionOutputDenseWeight.T(), layerParams.LayerAttentionOutputDenseBias, modelParams, ckksParams, eval)
	if err != nil {
		panic(err)
	}

	// step 5. residual connection
	ctResidual, err := matrix.CiphertextMatricesAddCiphertextMatrices(ctDense, ctInput, eval)
	if err != nil {
		panic(fmt.Sprintf("failed to add ciphertext matrices: %v", err))
	}

	ctResidualdec, err := he.DecryptCiphertextMatrices(ctResidual, modelParams, ckksParams, dec, ecd)
	if err != nil {
		panic(err)
	}

	ctResidualdecd := ecdmat.DecodeDense(ctResidualdec, modelParams)
	fmt.Println("ctResidualdecd:")
	utils.PrintMat(ctResidualdecd[0])

	// fmt.Println("Already residual connection")
	// Step 6. 计算layer norm
	ctLayernorm, err := layernorm.LayerNormSelfAttentionOutputTest(ctResidual, layerParams.LayerAttentionOutputLayerNormWeight, layerParams.LayerAttentionOutputLayerNormBias, modelParams, ckksParams, ecd, eval, dec)
	if err != nil {
		panic(fmt.Sprintf("failed to compute layer norm: %v", err))
	}

	fmt.Println("ctLayernorm level:", ctLayernorm.Ciphertexts[0].Level())

	ctQKTdec, err := he.DecryptCiphertextMatrices(ctLayernorm, modelParams, ckksParams, dec, ecd)
	if err != nil {
		panic(err)
	}

	ctQKTdecd := ecdmat.DecodeDense(ctQKTdec, modelParams)
	utils.PrintMat(ctQKTdecd[0])

	fmt.Println("SelfOutput computation finished.")
}

func CiphertextInferenceBertTinyIntermediate() {
	// This function is a placeholder for the Bert model ciphertext inference logic.
	// It should be implemented with the actual logic for performing inference on the Bert model.
	fmt.Println("----------Task: BERT Model Inference Under The CKKS Scheme Intermediate----------")

	// Step 1. Generate the model parameters and configurations
	ckksParams, _, modelParams, err := configs.InitBertTiny("short", 128, "bert_tiny_data_rte")
	if err != nil {
		panic(err)
	}

	// Step 2.1 Read the input data from the CSV file
	selfInputPath := filepath.Join(modelParams.ModelPath, "layer_0", "Intermediate", "allresults", "intermediate_inputs.csv")
	mat, err := utils.ReadCSVToMatrix(selfInputPath)
	if err != nil {
		panic(err)
	}
	row, col := mat.Dims()
	fmt.Printf("Input matrix dimensions: %d rows, %d columns\n", row, col)
	// Pad or truncate the matrix to match the model parameters
	mat = utils.PadOrTruncateMatrix(mat, modelParams)
	row, col = mat.Dims()
	fmt.Printf("Pad matrix dimensions: %d rows, %d columns\n", row, col)
	// Generate the input matrices for Bert model inference
	inputMatrices := utils.MatrixToBatchMats(mat, modelParams)

	// Step 2.2 Read the layer parameters for the first layer
	layerParams, err := utils.ReadLayerParameters(modelParams, 0)
	if err != nil {
		panic(fmt.Sprintf("failed to read layer parameters: %v", err))
	}
	utils.PrintLayerParametersDims(layerParams)

	// Step 3.1 Encode the input matrices
	encodedInputs := ecdmat.EncodeDense(inputMatrices, modelParams)
	// utils.PrintFirstOfVectors(encodedInputs, 10)
	decodedInputs := ecdmat.DecodeDense(encodedInputs, modelParams)
	utils.PrintMat(decodedInputs[0])

	// Step 3.2 Generate the keys for the CKKS scheme
	ecd, enc, eval, dec := he.GenerateKeys(ckksParams, modelParams)
	_ = dec
	// Step 3.3 Encrypt the input matrices
	ctX, err := he.EncryptInputMatrices(encodedInputs, modelParams, ckksParams, enc, ecd)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt input matrices: %v", err))
	}

	// Intermediate
	ctIntermediate, err := matrix.CiphertextMatricesMultiplyWeightAndAddBias(ctX, layerParams.LayerIntermediateDenseWeight.T(), layerParams.LayerIntermediateDenseBias, modelParams, ckksParams, eval)
	if err != nil {
		panic(err)
	}

	// GeLU
	ctGeLU := gelu.CiphertextMatricesGeluChebyshev(ctIntermediate, &ckksParams, eval, modelParams.GeluMinValue, modelParams.GeluMaxValue, modelParams.GeluDegree)

	// decryption
	ptDec, err := he.DecryptCiphertextMatrices(ctGeLU, modelParams, ckksParams, dec, ecd)
	if err != nil {
		panic(err)
	}

	ctDec := ecdmat.DecodeDense(ptDec, modelParams)
	utils.PrintMat(ctDec[0])

	fmt.Println("Intermediate computation finished.")
}

func CiphertextInferenceBertTinyOutput() {
	// This function is a placeholder for the Bert model ciphertext inference logic.
	// It should be implemented with the actual logic for performing inference on the Bert model.
	fmt.Println("----------Task: BERT Model Inference Under The CKKS Scheme Output----------")

	// Step 1. Generate the model parameters and configurations
	ckksParams, _, modelParams, err := configs.InitBertTiny("short", 128, "bert_tiny_data_rte")
	if err != nil {
		panic(err)
	}

	// Step 2.1 Read the input data from the CSV file
	SelfLayernormPath := filepath.Join(modelParams.ModelPath, "layer_0", "Attention", "SelfOutput", "allresults", "real_self_output.csv")
	matSelfLayernorm, err := utils.ReadCSVToMatrix(SelfLayernormPath)
	if err != nil {
		panic(err)
	}
	row, col := matSelfLayernorm.Dims()
	fmt.Printf("Input matrix dimensions: %d rows, %d columns\n", row, col)
	// Pad or truncate the matrix to match the model parameters
	matSelfLayernorm = utils.PadOrTruncateMatrix(matSelfLayernorm, modelParams)
	row, col = matSelfLayernorm.Dims()
	fmt.Printf("Pad matrix dimensions: %d rows, %d columns\n", row, col)
	// Generate the input matrices for Bert model inference
	selfLayernormMatrices := utils.MatrixToBatchMats(matSelfLayernorm, modelParams)

	// Step 2.1 Read the input data from the CSV file
	outputInputPath := filepath.Join(modelParams.ModelPath, "layer_0", "Output", "allresults", "final_output_inputs.csv")
	mat, err := utils.ReadCSVToMatrix(outputInputPath)
	if err != nil {
		panic(err)
	}
	row, col = mat.Dims()
	fmt.Printf("Input matrix dimensions: %d rows, %d columns\n", row, col)
	// Pad or truncate the matrix to match the model parameters
	mat = utils.PadOrTruncateMatrix(mat, modelParams)
	row, col = mat.Dims()
	fmt.Printf("Pad matrix dimensions: %d rows, %d columns\n", row, col)
	// Generate the input matrices for Bert model inference
	outputInputMatrices := utils.MatrixToBatchMats(mat, modelParams)

	// Step 2.2 Read the layer parameters for the first layer
	layerParams, err := utils.ReadLayerParameters(modelParams, 0)
	if err != nil {
		panic(fmt.Sprintf("failed to read layer parameters: %v", err))
	}
	utils.PrintLayerParametersDims(layerParams)

	// Step 3.1 Encode the input matrices
	encodedSelfLayernormMatrices := ecdmat.EncodeDense(selfLayernormMatrices, modelParams)
	encodedOutputInputs := ecdmat.EncodeDense(outputInputMatrices, modelParams)
	// utils.PrintFirstOfVectors(encodedInputs, 10)
	decodedInputs := ecdmat.DecodeDense(encodedOutputInputs, modelParams)
	fmt.Println("Decoded input matrices:")
	utils.PrintMat(decodedInputs[0])

	// Step 3.2 Generate the keys for the CKKS scheme
	ecd, enc, eval, dec := he.GenerateKeys(ckksParams, modelParams)
	_ = dec
	// Step 3.3 Encrypt the input matrices
	ctX, err := he.EncryptInputMatrices(encodedOutputInputs, modelParams, ckksParams, enc, ecd)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt input matrices: %v", err))
	}

	ctSelfOutputLayernorm, err := he.EncryptInputMatrices(encodedSelfLayernormMatrices, modelParams, ckksParams, enc, ecd)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt input matrices: %v", err))
	}

	// Step 4. 计算dense
	ctOutputDense, err := matrix.CiphertextMatricesMultiplyWeightAndAddBias(ctX, layerParams.LayerOutputDenseWeight.T(), layerParams.LayerOutputDenseBias, modelParams, ckksParams, eval)
	if err != nil {
		panic(err)
	}

	// step 5. residual connection
	ctResidual, err := matrix.CiphertextMatricesAddCiphertextMatrices(ctOutputDense, ctSelfOutputLayernorm, eval)
	if err != nil {
		panic(fmt.Sprintf("failed to add ciphertext matrices: %v", err))
	}

	// Step 6. 计算layer norm
	ctLayernorm, err := layernorm.LayerNormOutput(ctResidual, layerParams.LayerOutputLayerNormWeight, layerParams.LayerOutputLayerNormBias, modelParams, ckksParams, ecd, eval)
	if err != nil {
		panic(fmt.Sprintf("failed to compute layer norm: %v", err))
	}

	ptDec, err := he.DecryptCiphertextMatrices(ctLayernorm, modelParams, ckksParams, dec, ecd)
	if err != nil {
		panic(err)
	}

	ptDecode := ecdmat.DecodeDense(ptDec, modelParams)
	utils.PrintMat(ptDecode[0])

	fmt.Println("Output computation finished.")
}

func TestOperation() {
	fmt.Println("----------Task: Testing Operation ----------")

	// Step 1. Generate the model parameters and configurations
	ckksParams, btpParams, modelParams, err := configs.InitBert("base", 8, "bert_base_data")
	if err != nil {
		panic(err)
	}

	// Step 2. Generate Keys and Bootstrapping Evaluator
	ecd, enc, eval, dec, btpEval := he.GenerateKeysAndBtsKeys(ckksParams, btpParams, modelParams)

	// Step 3. Encode and Encrypt Two Plain Vectors (使用预分配 plaintext)
	slots := ckksParams.MaxSlots()
	values1 := make([]float64, slots)
	values2 := make([]float64, slots)
	for i := 0; i < slots; i++ {
		values1[i] = float64(i + 1)
		values2[i] = float64((i + 1) * 2)
	}

	// 创建可复用 plaintext 容器
	pt := ckks.NewPlaintext(ckksParams, ckksParams.MaxLevel())

	// 编码并加密 values1
	if err := ecd.Encode(values1, pt); err != nil {
		panic(err)
	}
	ct1, err := enc.EncryptNew(pt)
	if err != nil {
		panic(err)
	}

	// 编码并加密 values2
	if err := ecd.Encode(values2, pt); err != nil {
		panic(err)
	}
	ct2, err := enc.EncryptNew(pt)
	if err != nil {
		panic(err)
	}

	const iters = 100

	// -------------------- Add --------------------
	start := time.Now()
	for i := 0; i < iters; i++ {
		_, _ = eval.AddNew(ct1, ct2)
	}
	elapsed := time.Since(start)
	fmt.Printf("Add × %d: Total = %v, Avg = %v\n", iters, elapsed, elapsed/time.Duration(iters))

	// -------------------- Mul --------------------
	start = time.Now()
	for i := 0; i < iters; i++ {
		ctMul, _ := eval.MulRelinNew(ct1, ct2)
		eval.Rescale(ctMul, ctMul)
	}
	elapsed = time.Since(start)
	fmt.Printf("Mul × %d: Total = %v, Avg = %v\n", iters, elapsed, elapsed/time.Duration(iters))

	// -------------------- CMul --------------------
	start = time.Now()
	for i := 0; i < iters; i++ {
		_, _ = eval.MulNew(ct1, 3.0)
	}
	elapsed = time.Since(start)
	fmt.Printf("CMul × %d: Total = %v, Avg = %v\n", iters, elapsed, elapsed/time.Duration(iters))

	// -------------------- Rotate --------------------
	start = time.Now()
	for i := 0; i < iters; i++ {
		_, _ = eval.RotateNew(ct1, 1)
	}
	elapsed = time.Since(start)
	fmt.Printf("Rot × %d: Total = %v, Avg = %v\n", iters, elapsed, elapsed/time.Duration(iters))

	// -------------------- Bootstrap --------------------
	start = time.Now()
	for i := 0; i < iters; i++ {
		_, err := btpEval.Bootstrap(ct1)
		if err != nil {
			panic(err)
		}
	}
	elapsed = time.Since(start)
	fmt.Printf("Bts × %d: Total = %v, Avg = %v\n", iters, elapsed, elapsed/time.Duration(iters))

	// -------------------- Decryption for verification --------------------
	ctAdd, _ := eval.AddNew(ct1, ct2)
	decRes := make([]float64, slots)
	_ = ecd.Decode(dec.DecryptNew(ctAdd), decRes)
	fmt.Printf("Decrypted Add Result (First 10 values): %.4f\n", decRes[:10])
}

func CiphertextInferenceLlamaAttention(paramType string, rowValue int, modelPath string) {
	fmt.Println("----------Task: Llama Model Inference Under The CKKS Scheme Attention----------")

	// Step 1. Generate the model parameters and configurations
	ckksParams, btpParams, modelParams, err := configs.InitLlama(paramType, rowValue, modelPath)
	if err != nil {
		panic(err)
	}

	_ = ckksParams
	_ = btpParams

	// Step 2.1 Read the input data from the CSV file
	layerParams, mat := utils.GenerateLlamaLayerParametersAttention(modelParams)
	plainOutput := utils.ComputeLlamaAttentionPlain(layerParams, mat)
	fmt.Println("Plain Attention Output:")
	utils.PrintMat(plainOutput)

	fmt.Println("Llama Attention layer parameters and input matrix generated.", modelParams.NumBatch)
	row, col := mat.Dims()
	fmt.Printf("Input matrix dimensions: %d rows, %d columns\n", row, col)
	utils.PrintMat(mat)
	// Pad or truncate the matrix to match the model parameters
	mat = utils.PadOrTruncateMatrix(mat, modelParams)
	row, col = mat.Dims()
	fmt.Printf("Pad matrix dimensions: %d rows, %d columns\n", row, col)
	// Generate the input matrices for Bert model inference
	inputMatrices := utils.MatrixToBatchMats(mat, modelParams)

	// Step 3.1 Encode the input matrices
	encodedInputs := ecdmat.EncodeDense(inputMatrices, modelParams)
	// utils.PrintFirstOfVectors(encodedInputs, 10)
	decodedInputs := ecdmat.DecodeDense(encodedInputs, modelParams)
	utils.PrintMat(decodedInputs[0])

	// Step 3.2 Generate the keys for the CKKS scheme
	ecd, enc, eval, dec, btpEval := he.GenerateKeysAndBtsKeys(ckksParams, btpParams, modelParams)
	_ = dec
	_ = btpEval
	_ = eval
	// Step 3.3 Encrypt the input matrices
	ctX, err := he.EncryptInputMatrices(encodedInputs, modelParams, ckksParams, enc, ecd)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt input matrices: %v", err))
	}
	_ = ctX

	// vecCos, vecSin := matrix.GenRoPEBroadcast(128, 8, 4, 10000.0)
	// _ = vecCos
	// _ = vecSin

	// Step 4. 计算Q、K、V
	ctQ, ctK, ctV, err := llama.ComputeQKVLlama(ctX, layerParams, modelParams, ckksParams, eval)
	if err != nil {
		panic(fmt.Sprintf("failed to compute QKV: %v", err))
	}

	ctAttention, err := llama.ComputeAttentionLlama(ctQ, ctK, ctV, modelParams, &ckksParams, ecd, enc, eval, btpEval)

	// ctQsplit := matrix.SplitCiphertextMatricesByNumber(ctQ, modelParams.GroupQ)
	// ctKsplit := matrix.SplitCiphertextMatricesByNumber(ctK, modelParams.GroupKV)
	// ctVsplit := matrix.SplitCiphertextMatricesByNumber(ctV, modelParams.GroupKV)

	// // Q 和 K 应用 RoPE
	// ctQsplitRoPE := matrix.CiphertextMatricesComputeRotaryPositionEmbedding(ctQsplit, modelParams, ckksParams, eval)
	// ctKsplitRoPE := matrix.CiphertextMatricesComputeRotaryPositionEmbedding(ctKsplit, modelParams, ckksParams, eval)
	// // _ = ctQsplitRoPE
	// ctAttentionHeads := make([]*he.CiphertextMatrices, modelParams.NumHeads)

	// // 3.2) For each head, compute attention
	// stepSizeQ := modelParams.GroupQ / modelParams.GroupKV
	// for j := 0; j < modelParams.GroupKV; j++ {
	// 	for k := 0; k < stepSizeQ; k++ {
	// 		ctAttention, ctAdd, err := bert.ComputeAttention(ctQsplitRoPE[j*stepSizeQ+k], ctKsplitRoPE[j], ctVsplit[j], modelParams, &ckksParams, ecd, enc, eval, btpEval)
	// 		if err != nil {
	// 			panic(fmt.Sprintf("failed to compute attention for head %d i: %v", j, err))
	// 		}
	// 		_ = ctAdd // ctAdd is not used, but computed for consistency
	// 		ctAttentionHeads[j*stepSizeQ+k] = ctAttention
	// 		// fmt.Printf("Layer %d, Head %d: Computed attention with %d ciphertexts\n", i, j, len(ctAttention.Ciphertexts))
	// 	}
	// }

	// // 3.3) Combine the attention heads
	// ctAttention := matrix.MergeCiphertextMatricesByHeads(ctAttentionHeads)

	ctAttentionDec, err := he.DecryptCiphertextMatrices(ctAttention, modelParams, ckksParams, dec, ecd)
	if err != nil {
		panic(err)
	}

	ctQRoPEDecd := ecdmat.DecodeDense(ctAttentionDec, modelParams)
	utils.PrintMat(ctQRoPEDecd[0])
	// fmt.Println(len(ctQsplit), len(ctKsplit), len(ctVsplit))

}

func CiphertextInferenceLlamaTest(paramType string, rowValue int, modelPath string) {
	fmt.Println("----------Task: Llama Model Inference Under The CKKS Scheme Attention----------")

	// Step 1. Generate the model parameters and configurations
	ckksParams, btpParams, modelParams, err := configs.InitLlama(paramType, rowValue, modelPath)
	if err != nil {
		panic(err)
	}

	// Step 2.1 Read the input data from the CSV file
	mat, err := utils.ReadLlamaInput(modelParams, "input_rmsnorm_inputs.csv")
	if err != nil {
		panic(err)
	}
	row, col := mat.Dims()
	fmt.Printf("Input matrix dimensions: %d rows, %d columns\n", row, col)
	// Pad or truncate the matrix to match the model parameters
	mat = utils.PadOrTruncateMatrix(mat, modelParams)
	row, col = mat.Dims()
	fmt.Printf("Pad matrix dimensions: %d rows, %d columns\n", row, col)
	// Generate the input matrices for Bert model inference
	inputMatrices := utils.MatrixToBatchMats(mat, modelParams)

	// Step 2.2 Read the layer parameters for the first layer
	layerParams, err := utils.ReadLlamaLayerParameters(modelParams, 0)
	if err != nil {
		panic(fmt.Sprintf("failed to read layer parameters: %v", err))
	}
	utils.PrintLlamaLayerParametersDims(layerParams)

	// Step 3.1 Encode the input matrices
	encodedInputs := ecdmat.EncodeDense(inputMatrices, modelParams)
	// utils.PrintFirstOfVectors(encodedInputs, 10)
	decodedInputs := ecdmat.DecodeDense(encodedInputs, modelParams)
	utils.PrintMat(decodedInputs[0])

	// Step 3.2 Generate the keys for the CKKS scheme
	ecd, enc, eval, dec, btpEval := he.GenerateKeysAndBtsKeys(ckksParams, btpParams, modelParams)
	_ = dec
	_ = btpEval
	// Step 3.3 Encrypt the input matrices
	ctX, err := he.EncryptInputMatrices(encodedInputs, modelParams, ckksParams, enc, ecd)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt input matrices: %v", err))
	}

	// Step 4 Attention
	// Step 4.1 RMSNormInput
	ctInputRMSNorm, err := rmsnorm.RMSNormSelfAttentionInput(ctX, layerParams.InputRmsNormWeight, modelParams, ckksParams, ecd, eval)
	if err != nil {
		panic(err)
	}

	fmt.Println("Before level:", ctInputRMSNorm.Ciphertexts[0].Level())
	// 4.2 bts
	ctAttentionInput, err := btp.CiphertextMatricesBootstrappingMT(ctInputRMSNorm, btpEval, btpParams, 24)
	if err != nil {
		panic(err)
	}
	fmt.Println("After level:", ctAttentionInput.Ciphertexts[0].Level())

	// Step 4.3 计算Q、K、V
	ctQ, ctK, ctV, err := llama.ComputeQKVLlama(ctAttentionInput, layerParams, modelParams, ckksParams, eval)
	if err != nil {
		panic(fmt.Sprintf("failed to compute QKV: %v", err))
	}

	//dec
	ctAttentiondec, err := he.DecryptCiphertextMatrices(ctK, modelParams, ckksParams, dec, ecd)
	if err != nil {
		panic(err)
	}

	ctAttentiondecd := ecdmat.DecodeDense(ctAttentiondec, modelParams)
	utils.PrintMat(ctAttentiondecd[0])

	// Step4.4 compute selfAttention
	ctAttentionConcate, err := llama.ComputeAttentionLlama(ctQ, ctK, ctV, modelParams, &ckksParams, ecd, enc, eval, btpEval)
	if err != nil {
		panic(err)
	}

	// step4.5 bts
	ctAttentionConcateBts, err := btp.CiphertextMatricesBootstrappingMT(ctAttentionConcate, btpEval, btpParams, 24)
	if err != nil {
		panic(err)
	}

	// step4.6 concate
	ctAttention, err := matrix.CiphertextMatricesMultiplyPlaintextMatrix(ctAttentionConcateBts, modelParams, layerParams.LayerConcateWeight.T(), ckksParams, eval)
	if err != nil {
		panic(err)
	}

	// step 4.7 residual
	ctResidualAttention, err := matrix.CiphertextMatricesAddCiphertextMatrices(ctAttention, ctAttentionInput, eval)
	if err != nil {
		panic(fmt.Sprintf("failed to add ciphertext matrices: %v", err))
	}

	// 4.8 FFN RMS
	ctFFNInputRMSNorm, err := rmsnorm.RMSNormFFNInput(ctResidualAttention, layerParams.InputRmsNormWeight, modelParams, ckksParams, ecd, eval)
	if err != nil {
		panic(err)
	}

	// step4.9 bts
	ctFFNInputRMSNormBts, err := btp.CiphertextMatricesBootstrappingMT(ctFFNInputRMSNorm, btpEval, btpParams, 24)
	if err != nil {
		panic(err)
	}

	// step4.10 FFN
	ctFFNOutput, err := llama.ComputeFFNLlama(ctFFNInputRMSNormBts, layerParams, modelParams, ckksParams, eval)
	if err != nil {
		panic(err)
	}

	// step 4.11 residual
	ctOutput, err := matrix.CiphertextMatricesAddCiphertextMatrices(ctResidualAttention, ctFFNOutput, eval)
	if err != nil {
		panic(fmt.Sprintf("failed to add ciphertext matrices: %v", err))
	}

	//dec
	ctQKTdec, err := he.DecryptCiphertextMatrices(ctOutput, modelParams, ckksParams, dec, ecd)
	if err != nil {
		panic(err)
	}

	ctQKTdecd := ecdmat.DecodeDense(ctQKTdec, modelParams)
	utils.PrintMat(ctQKTdecd[0])

}
