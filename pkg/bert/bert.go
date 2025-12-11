package bert

import (
	"Arion/configs"
	"Arion/pkg/btp"
	"Arion/pkg/ecdmat"
	"Arion/pkg/he"
	"Arion/pkg/math/activation/gelu"
	"Arion/pkg/math/activation/layernorm"
	"Arion/pkg/math/activation/softmax"
	"Arion/pkg/math/activation/tanh"
	"Arion/pkg/math/matrix"
	"Arion/pkg/utils"
	"fmt"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// InferenceBert performs the BERT model inference under the CKKS scheme.
// It initializes the model parameters, reads input data, encrypts it, and performs the inference
// through multiple layers of attention mechanisms.
// paramType specifies the type of parameters to use, e.g., "short", "base".
// It prints the progress and results of each layer's attention computation.
// Note: This function assumes that the necessary packages and configurations are correctly set up.
func InferenceBert(paramType string, rowValue int, modelPath string) {

	fmt.Print("----------Task: BERT Model Inference Under The CKKS Scheme " + paramType + "----------\n")

	/* Group1: Generate the model parameters, configurations and data */
	// 1) Generate the model parameters and configurations
	ckksParams, btpParams, modelParams, err := configs.InitBert(paramType, rowValue, modelPath)
	if err != nil {
		panic(err)
	}
	outputFilePath := "Output/" + modelParams.ModelPath

	// 2.1) Read the input data from the CSV file
	mat, err := utils.ReadInput(modelParams, "embedded_inputs.csv")
	if err != nil {
		panic("failed to read input: " + err.Error())
	}
	mat = utils.PadOrTruncateMatrix(mat, modelParams)
	// 2.2) Encode and batch the input matrix
	inputMats := utils.MatrixToBatchMats(mat, modelParams)

	// 3) Generate keys and bootstrapping keys
	ecd, enc, eval, dec, btpEval := he.GenerateKeysAndBtsKeys(ckksParams, btpParams, modelParams)

	// 4) Encrypt the input matrices
	encodedInputs := ecdmat.EncodeDense(inputMats, modelParams)
	ctX, err := he.EncryptInputMatrices(encodedInputs, modelParams, ckksParams, enc, ecd)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt input matrices: %v", err))
	}

	/* Perform the Bert model inference */
	for i := 0; i < modelParams.NumLayers; i++ {
		fmt.Printf("================== Layer %d  ==================\n", i)
		/* Group2: Perform the Bert model Attention inference  */
		// 1) Read the layer parameters for the current layer
		layerParams, err := utils.ReadLayerParameters(modelParams, i)
		if err != nil {
			panic(fmt.Sprintf("failed to read layer parameters for layer %d: %v", i, err))
		}
		// utils.PrintLayerParametersDims(layerParams)

		// 2) Compute Q, K, V matrices
		ctQ, ctK, ctV, err := ComputeQKV(ctX, layerParams, modelParams, ckksParams, eval)
		if err != nil {
			panic(fmt.Sprintf("failed to compute QKV matrices: %v", err))
		}

		// 3.1) Split the ciphertext matrices by heads
		ctQsplit := matrix.SplitCiphertextMatricesByHeads(ctQ, modelParams)
		ctKsplit := matrix.SplitCiphertextMatricesByHeads(ctK, modelParams)
		ctVsplit := matrix.SplitCiphertextMatricesByHeads(ctV, modelParams)
		ctAttentionHeads := make([]*he.CiphertextMatrices, modelParams.NumHeads)
		// 3.2) For each head, compute attention
		for j := 0; j < modelParams.NumHeads; j++ {
			ctAttention, ctAdd, err := ComputeAttention(ctQsplit[j], ctKsplit[j], ctVsplit[j], modelParams, &ckksParams, ecd, enc, eval, btpEval)
			if err != nil {
				panic(fmt.Sprintf("failed to compute attention for head %d in layer %d: %v", j, i, err))
			}
			_ = ctAdd // ctAdd is not used, but computed for consistency
			ctAttentionHeads[j] = ctAttention
			fmt.Printf("Layer %d, Head %d: Computed attention with %d ciphertexts\n", i, j, len(ctAttention.Ciphertexts))
		}
		// 3.3) Combine the attention heads
		ctAttention := matrix.MergeCiphertextMatricesByHeads(ctAttentionHeads)

		fmt.Println("Attention computation finished")

		/* **Bootstrapping 1** */
		ctAttentionOutput, err := btp.CiphertextMatricesBootstrapping(ctAttention, btpEval)
		if err != nil {
			panic(fmt.Sprintf("failed to bootstrap attention ciphertexts: %v", err))
		}

		fmt.Println("Bootstrapping for attention finished")
		fmt.Println("After Bootstrapping Level:", ctAttentionOutput.Ciphertexts[0].Level())

		/* Group3: Perform the Bert model Attention SelfOutput inference  */
		ctSelfOutputDense, err := matrix.CiphertextMatricesMultiplyWeightAndAddBias(ctAttentionOutput, layerParams.LayerAttentionOutputDenseWeight.T(), layerParams.LayerAttentionOutputDenseBias, modelParams, ckksParams, eval)
		if err != nil {
			panic(err)
		}
		ctSelfOutputResidual, err := matrix.CiphertextMatricesAddCiphertextMatrices(ctSelfOutputDense, ctX, eval)
		if err != nil {
			panic(err)
		}

		ctSelfOutputLayernorm, err := layernorm.LayerNormSelfAttentionOutput(ctSelfOutputResidual, layerParams.LayerAttentionOutputLayerNormWeight, layerParams.LayerAttentionOutputLayerNormBias, modelParams, ckksParams, ecd, eval)
		if err != nil {
			panic(err)
		}
		fmt.Println("SelfOutputLayernorm computation finished")

		/* **Bootstrapping 2** */
		ctSelfOutput, err := btp.CiphertextMatricesBootstrapping(ctSelfOutputLayernorm, btpEval)
		if err != nil {
			panic(err)
		}
		fmt.Println("Bootstrapping for SelfOutputLayernorm finished")
		fmt.Println("After Bootstrapping Level:", ctSelfOutput.Ciphertexts[0].Level())

		/* Group4: Perform the Bert model Intermediate inference  */
		// Intermediate
		ctIntermediate, err := matrix.CiphertextMatricesMultiplyWeightAndAddBias(ctSelfOutput, layerParams.LayerIntermediateDenseWeight.T(), layerParams.LayerIntermediateDenseBias, modelParams, ckksParams, eval)
		if err != nil {
			panic(err)
		}

		// GeLU
		ctGeLU := gelu.CiphertextMatricesGeluChebyshev(ctIntermediate, &ckksParams, eval, modelParams.GeluMinValue, modelParams.GeluMaxValue, modelParams.GeluDegree)

		// Linear Output
		ctOutputDense, err := matrix.CiphertextMatricesMultiplyWeightAndAddBias(ctGeLU, layerParams.LayerOutputDenseWeight.T(), layerParams.LayerOutputDenseBias, modelParams, ckksParams, eval)
		if err != nil {
			panic(err)
		}

		fmt.Println("Intermediate computation finished")

		/* **Bootstrapping 3** */
		ctOutputLayernormInput, err := btp.CiphertextMatricesBootstrapping(ctOutputDense, btpEval)
		if err != nil {
			panic(err)
		}
		fmt.Println("Bootstrapping for Intermediate finished")
		fmt.Println("After Bootstrapping Level:", ctOutputLayernormInput.Ciphertexts[0].Level())

		// residual connection
		ctOutputResidual, err := matrix.CiphertextMatricesAddCiphertextMatrices(ctOutputLayernormInput, ctSelfOutput, eval)
		if err != nil {
			panic(fmt.Sprintf("failed to add ciphertext matrices: %v", err))
		}

		// compute layernorm
		ctOutputLayernorm, err := layernorm.LayerNormOutput(ctOutputResidual, layerParams.LayerOutputLayerNormWeight, layerParams.LayerOutputLayerNormBias, modelParams, ckksParams, ecd, eval)
		if err != nil {
			panic(fmt.Sprintf("failed to compute layer norm: %v", err))
		}

		fmt.Println("OutputLayernorm computation finished")

		/* **Bootstrapping 4** */
		ctOutput, err := btp.CiphertextMatricesBootstrapping(ctOutputLayernorm, btpEval)
		if err != nil {
			panic(err)
		}

		fmt.Println("Bootstrapping for OutputLayernorm finished")
		fmt.Println("After Bootstrapping Level:", ctOutput.Ciphertexts[0].Level())

		fmt.Println("Layer", i, "inference result:")
		ptDec, err := he.DecryptCiphertextMatrices(ctOutput, modelParams, ckksParams, dec, ecd)
		if err != nil {
			panic(err)
		}

		ptDecode := ecdmat.DecodeDense(ptDec, modelParams)
		utils.PrintMat(ptDecode[0])

		// Save the output of each layer to a CSV file
		filePath := fmt.Sprintf(outputFilePath+"/layer_%d_output.csv", i)
		err = utils.SaveMatrixToCSV(ptDecode[0], filePath)
		if err != nil {
			fmt.Println("保存失败:", err)
		}

		ctX = ctOutput // Update ctX for the next layer
	}

	ptDec, err := he.DecryptCiphertextMatrices(ctX, modelParams, ckksParams, dec, ecd)
	if err != nil {
		panic(err)
	}

	ptDecode := ecdmat.DecodeDense(ptDec, modelParams)
	utils.PrintMat(ptDecode[0])
	fmt.Println("All Bert Inference computation finished.")
}

func InferenceBertTiny(paramType string, rowValue int, modelPath string) {

	fmt.Print("----------Task: BERT-Tiny Model Inference Under The CKKS Scheme " + paramType + "----------\n")

	/* Group1: Generate the model parameters, configurations and data */
	// 1) Generate the model parameters and configurations
	ckksParams, btpParams, modelParams, err := configs.InitBertTiny(paramType, rowValue, modelPath)
	if err != nil {
		panic(err)
	}

	outputFilePath := "Output/" + modelParams.ModelPath

	// 2.1) Read the input data from the CSV file
	mat, err := utils.ReadInput(modelParams, "embedded_inputs.csv")
	if err != nil {
		panic("failed to read input: " + err.Error())
	}
	mat = utils.PadOrTruncateMatrix(mat, modelParams)
	// 2.2) Encode and batch the input matrix
	inputMats := utils.MatrixToBatchMats(mat, modelParams)

	// 3) Generate keys and bootstrapping keys
	ecd, enc, eval, dec, btpEval := he.GenerateKeysAndBtsKeys(ckksParams, btpParams, modelParams)

	// 4) Encrypt the input matrices
	encodedInputs := ecdmat.EncodeDense(inputMats, modelParams)
	ctX, err := he.EncryptInputMatrices(encodedInputs, modelParams, ckksParams, enc, ecd)
	if err != nil {
		panic(fmt.Sprintf("failed to encrypt input matrices: %v", err))
	}

	/* Perform the Bert model inference */
	for i := 0; i < modelParams.NumLayers; i++ {
		fmt.Printf("================== Layer %d  ==================\n", i)
		/* Group2: Perform the Bert model Attention inference  */
		// 1) Read the layer parameters for the current layer
		layerParams, err := utils.ReadLayerParameters(modelParams, i)
		if err != nil {
			panic(fmt.Sprintf("failed to read layer parameters for layer %d: %v", i, err))
		}
		// utils.PrintLayerParametersDims(layerParams)

		// 2) Compute Q, K, V matrices
		ctQ, ctK, ctV, err := ComputeQKV(ctX, layerParams, modelParams, ckksParams, eval)
		if err != nil {
			panic(fmt.Sprintf("failed to compute QKV matrices: %v", err))
		}

		// 3.1) Split the ciphertext matrices by heads
		ctQsplit := matrix.SplitCiphertextMatricesByHeads(ctQ, modelParams)
		ctKsplit := matrix.SplitCiphertextMatricesByHeads(ctK, modelParams)
		ctVsplit := matrix.SplitCiphertextMatricesByHeads(ctV, modelParams)
		ctAttentionHeads := make([]*he.CiphertextMatrices, modelParams.NumHeads)
		// 3.2) For each head, compute attention
		for j := 0; j < modelParams.NumHeads; j++ {
			ctAttention, ctAdd, err := ComputeAttention(ctQsplit[j], ctKsplit[j], ctVsplit[j], modelParams, &ckksParams, ecd, enc, eval, btpEval)
			if err != nil {
				panic(fmt.Sprintf("failed to compute attention for head %d in layer %d: %v", j, i, err))
			}
			_ = ctAdd // ctAdd is not used, but computed for consistency
			ctAttentionHeads[j] = ctAttention
			// fmt.Printf("Layer %d, Head %d: Computed attention with %d ciphertexts\n", i, j, len(ctAttention.Ciphertexts))
		}
		// 3.3) Combine the attention heads
		ctAttention := matrix.MergeCiphertextMatricesByHeads(ctAttentionHeads)

		fmt.Println("Attention computation finished")

		/* **Bootstrapping 1** */
		ctAttentionOutput, err := btp.CiphertextMatricesBootstrapping(ctAttention, btpEval)
		if err != nil {
			panic(fmt.Sprintf("failed to bootstrap attention ciphertexts: %v", err))
		}

		fmt.Println("Bootstrapping for attention finished")
		fmt.Println("After Bootstrapping Level:", ctAttentionOutput.Ciphertexts[0].Level())

		/* Group3: Perform the Bert model Attention SelfOutput inference  */
		ctSelfOutputDense, err := matrix.CiphertextMatricesMultiplyWeightAndAddBias(ctAttentionOutput, layerParams.LayerAttentionOutputDenseWeight.T(), layerParams.LayerAttentionOutputDenseBias, modelParams, ckksParams, eval)
		if err != nil {
			panic(err)
		}
		ctSelfOutputResidual, err := matrix.CiphertextMatricesAddCiphertextMatrices(ctSelfOutputDense, ctX, eval)
		if err != nil {
			panic(err)
		}

		ctSelfOutputLayernorm, err := layernorm.LayerNormSelfAttentionOutput(ctSelfOutputResidual, layerParams.LayerAttentionOutputLayerNormWeight, layerParams.LayerAttentionOutputLayerNormBias, modelParams, ckksParams, ecd, eval)
		if err != nil {
			panic(err)
		}
		fmt.Println("SelfOutputLayernorm computation finished")

		/* **Bootstrapping 2** */
		ctSelfOutput, err := btp.CiphertextMatricesBootstrapping(ctSelfOutputLayernorm, btpEval)
		if err != nil {
			panic(err)
		}
		fmt.Println("Bootstrapping for SelfOutputLayernorm finished")
		fmt.Println("After Bootstrapping Level:", ctSelfOutput.Ciphertexts[0].Level())

		/* Group4: Perform the Bert model Intermediate inference  */
		// Intermediate
		ctIntermediate, err := matrix.CiphertextMatricesMultiplyWeightAndAddBias(ctSelfOutput, layerParams.LayerIntermediateDenseWeight.T(), layerParams.LayerIntermediateDenseBias, modelParams, ckksParams, eval)
		if err != nil {
			panic(err)
		}

		// GeLU
		ctGeLU := gelu.CiphertextMatricesGeluChebyshev(ctIntermediate, &ckksParams, eval, modelParams.GeluMinValue, modelParams.GeluMaxValue, modelParams.GeluDegree)

		// Linear Output
		ctOutputDense, err := matrix.CiphertextMatricesMultiplyWeightAndAddBias(ctGeLU, layerParams.LayerOutputDenseWeight.T(), layerParams.LayerOutputDenseBias, modelParams, ckksParams, eval)
		if err != nil {
			panic(err)
		}

		fmt.Println("Intermediate computation finished")

		/* **Bootstrapping 3** */
		ctOutputLayernormInput, err := btp.CiphertextMatricesBootstrapping(ctOutputDense, btpEval)
		if err != nil {
			panic(err)
		}
		fmt.Println("Bootstrapping for Intermediate finished")
		fmt.Println("After Bootstrapping Level:", ctOutputLayernormInput.Ciphertexts[0].Level())

		// residual connection
		ctOutputResidual, err := matrix.CiphertextMatricesAddCiphertextMatrices(ctOutputLayernormInput, ctSelfOutput, eval)
		if err != nil {
			panic(fmt.Sprintf("failed to add ciphertext matrices: %v", err))
		}

		// compute layernorm
		ctOutputLayernorm, err := layernorm.LayerNormOutput(ctOutputResidual, layerParams.LayerOutputLayerNormWeight, layerParams.LayerOutputLayerNormBias, modelParams, ckksParams, ecd, eval)
		if err != nil {
			panic(fmt.Sprintf("failed to compute layer norm: %v", err))
		}

		fmt.Println("OutputLayernorm computation finished")

		/* **Bootstrapping 4** */
		ctOutput, err := btp.CiphertextMatricesBootstrapping(ctOutputLayernorm, btpEval)
		if err != nil {
			panic(err)
		}

		fmt.Println("Bootstrapping for OutputLayernorm finished")
		fmt.Println("After Bootstrapping Level:", ctOutput.Ciphertexts[0].Level())

		// 显示信息
		fmt.Println("Layer", i, "inference result:")
		ptDec, err := he.DecryptCiphertextMatrices(ctOutput, modelParams, ckksParams, dec, ecd)
		if err != nil {
			panic(err)
		}

		ptDecode := ecdmat.DecodeDense(ptDec, modelParams)
		utils.PrintMat(ptDecode[0])
		// Save the output of each layer to a CSV file
		filePath := fmt.Sprintf(outputFilePath+"/layer_%d_output.csv", i)
		err = utils.SaveMatrixToCSV(ptDecode[0], filePath)
		if err != nil {
			fmt.Println("保存失败:", err)
		}
		ctX = ctOutput // Update ctX for the next layer
	}

	fmt.Printf("================== Pooler Output ==================\n")
	// Linear
	poolerDenseWeight, poolerDenseBias, err := utils.ReadPoolerParameters(modelParams)
	if err != nil {
		panic(fmt.Sprintf("failed to read pooler parameters: %v", err))
	}
	poolerLinearOutput, err := matrix.CiphertextMatricesMultiplyWeightAndAddBias(ctX, poolerDenseWeight.T(), poolerDenseBias, modelParams, ckksParams, eval)
	if err != nil {
		panic(err)
	}
	poolerLinearBeforeTanhOutput, err := tanh.CiphertextMatricesOutputLinear(poolerLinearOutput, modelParams, ckksParams, eval)
	if err != nil {
		panic(err)
	}

	//tanh
	poolerTanhOutput := tanh.CiphertextMatricesTanhChebyshev(poolerLinearBeforeTanhOutput, &ckksParams, eval)
	ptPoolerOutput, err := he.DecryptCiphertextMatrices(poolerTanhOutput, modelParams, ckksParams, dec, ecd)
	if err != nil {
		panic(err)
	}
	ptPoolerDecode := ecdmat.DecodeOutputDenseToMatrix(ptPoolerOutput, modelParams)
	fmt.Println("Pooler output:")
	utils.PrintMat(ptPoolerDecode)

	err = utils.SaveMatrixToCSV(ptPoolerDecode, outputFilePath+"/bert_output.csv")
	if err != nil {
		fmt.Println("保存失败:", err)
	}

	fmt.Println("All Bert Inference computation finished.")
}

// ComputeQKV 计算 Q、K、V 矩阵
// 输入：ctInput 为输入密文指针，layerParams 为权重和偏置参数指针
// 输出：Q、K、V 的密文表达指针
func ComputeQKV(
	ctInput *he.CiphertextMatrices,
	layerParams *utils.LayerParameters,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	eval *ckks.Evaluator,
) (*he.CiphertextMatrices, *he.CiphertextMatrices, *he.CiphertextMatrices, error) {
	// Q = ctInput × weightQ + biasQ
	ctQ, err := matrix.CiphertextMatricesMultiplyWeightAndAddBias(ctInput, layerParams.LayerAttentionSelfQueryWeight.T(), layerParams.LayerAttentionSelfQueryBias, modelParams, ckksParams, eval)
	if err != nil {
		return nil, nil, nil, err
	}
	// K = ctInput × weightK + biasK
	ctK, err := matrix.CiphertextMatricesMultiplyWeightAndAddBias(ctInput, layerParams.LayerAttentionSelfKeyWeight.T(), layerParams.LayerAttentionSelfKeyBias, modelParams, ckksParams, eval)
	if err != nil {
		return nil, nil, nil, err
	}
	// V = ctInput × weightV + biasV
	ctV, err := matrix.CiphertextMatricesMultiplyWeightAndAddBias(ctInput, layerParams.LayerAttentionSelfValueWeight.T(), layerParams.LayerAttentionSelfValueBias, modelParams, ckksParams, eval)
	if err != nil {
		return nil, nil, nil, err
	}
	return ctQ, ctK, ctV, nil
}

func ComputeAttention(
	ctQ, ctK, ctV *he.CiphertextMatrices,
	modelParams *configs.ModelParams,
	ckksParams *ckks.Parameters,
	ecd *ckks.Encoder,
	enc *rlwe.Encryptor,
	eval *ckks.Evaluator,
	btpEval *bootstrapping.Evaluator,
) (*he.CiphertextMatrices, *rlwe.Ciphertext, error) {
	//step.1 compute exp(x-modelParams.ExpSubValue)
	ctQKV, ctSum, err := matrix.CiphertextMatricesComputeAttentionWithBSGSAndApproxMax(ctQ, ctK, ctV, modelParams, *ckksParams, eval)
	if err != nil {
		panic("failed to compute exp(x-modelParams.ExpSubValue): " + err.Error())
	}

	ctSumBts, err := btpEval.Bootstrap(ctSum)
	if err != nil {
		panic(err)
	}
	//step.2 compute 1/sum(exp(x-modelParams.ExpSubValue))
	ctQKTInvSum := softmax.CiphertextInverse(ctSumBts, ckksParams, eval, modelParams.InvMinValue, modelParams.InvMaxValue, modelParams.InvDegree, modelParams.InvIter)

	// step.3 compute QKV * 1/sum(exp(x-modelParams.ExpSubValue)))
	newCiphertexts := make([]*rlwe.Ciphertext, ctQKV.NumCol)
	for i := 0; i < ctQKV.NumCol; i++ {
		ct, err := eval.MulRelinNew(ctQKV.Ciphertexts[i], ctQKTInvSum)
		if err != nil {
			panic("failed to multiply QKV with inverse sum: " + err.Error())
		}
		eval.Rescale(ct, ct)
		newCiphertexts[i] = ct
	}

	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumRow:      ctQKV.NumRow,
		NumCol:      ctQKV.NumCol,
		NumBatch:    ctQKV.NumBatch,
	}, ctSumBts, nil
	// return ctQKV, ctQKTInvSum, nil
}
