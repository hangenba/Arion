package llama

import (
	"Arion/configs"
	"Arion/pkg/he"
	"Arion/pkg/math/activation/silu"
	"Arion/pkg/math/activation/softmax"
	"Arion/pkg/math/matrix"
	"Arion/pkg/utils"
	"fmt"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// ComputeQKV 计算 Q、K、V 矩阵
// 输入：ctInput 为输入密文指针，layerParams 为权重和偏置参数指针
// 输出：Q、K、V 的密文表达指针
func ComputeQKVLlama(
	ctInput *he.CiphertextMatrices,
	layerParams *utils.LlamaLayerParameters,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	eval *ckks.Evaluator,
) (*he.CiphertextMatrices, *he.CiphertextMatrices, *he.CiphertextMatrices, error) {
	// Q = ctInput × weightQ
	ctQ, err := matrix.CiphertextMatricesMultiplyPlaintextMatrix(ctInput, modelParams, layerParams.LayerAttentionSelfQueryWeight.T(), ckksParams, eval)
	if err != nil {
		return nil, nil, nil, err
	}
	// K = ctInput × weightK
	ctK, err := matrix.CiphertextMatricesMultiplyPlaintextMatrix(ctInput, modelParams, layerParams.LayerAttentionSelfKeyWeight.T(), ckksParams, eval)
	if err != nil {
		return nil, nil, nil, err
	}
	// V = ctInput × weightV
	ctV, err := matrix.CiphertextMatricesMultiplyPlaintextMatrix(ctInput, modelParams, layerParams.LayerAttentionSelfValueWeight.T(), ckksParams, eval)
	if err != nil {
		return nil, nil, nil, err
	}
	return ctQ, ctK, ctV, nil
}

// 由于llama的attention计算和bert不一样，单独写一个ComputeAttention函数
// 具体计算中ctQ的维度32，ctK的维度8，ctV的维度8
// 4个Q对应1个K和1个V
func ComputeAttentionLlama(
	ctQ, ctK, ctV *he.CiphertextMatrices,
	modelParams *configs.ModelParams,
	ckksParams *ckks.Parameters,
	ecd *ckks.Encoder,
	enc *rlwe.Encryptor,
	eval *ckks.Evaluator,
	btpEval *bootstrapping.Evaluator,
) (*he.CiphertextMatrices, error) {
	ctQsplit := matrix.SplitCiphertextMatricesByNumber(ctQ, modelParams.GroupQ)
	ctKsplit := matrix.SplitCiphertextMatricesByNumber(ctK, modelParams.GroupKV)
	ctVsplit := matrix.SplitCiphertextMatricesByNumber(ctV, modelParams.GroupKV)

	// Q 和 K 应用 RoPE
	ctQsplitRoPE := matrix.CiphertextMatricesComputeRotaryPositionEmbedding(ctQsplit, modelParams, *ckksParams, eval)
	ctKsplitRoPE := matrix.CiphertextMatricesComputeRotaryPositionEmbedding(ctKsplit, modelParams, *ckksParams, eval)
	// _ = ctQsplitRoPE
	ctAttentionHeads := make([]*he.CiphertextMatrices, modelParams.NumHeads)

	babyStep := modelParams.BabyStep
	baseLen := modelParams.NumBatch

	// 3.2) For each head, compute attention
	stepSizeQ := modelParams.GroupQ / modelParams.GroupKV
	for j := 0; j < modelParams.GroupKV; j++ {
		for k := 0; k < stepSizeQ; k++ {

			babySteps := make([]int, 0, babyStep)
			for i := 0; i < babyStep; i++ {
				babySteps = append(babySteps, i*baseLen)
			}

			ctKRotated := matrix.RotateCiphertextMatricesHoisting(ctKsplitRoPE[j], babySteps, eval)
			ctVRotated := matrix.RotateCiphertextMatricesHoisting(ctVsplit[j], babySteps, eval)

			ctAttention, ctAdd, err := ComputeGroupAttentionLlama(ctQsplitRoPE[j*stepSizeQ+k], ctKRotated, ctVRotated, modelParams, ckksParams, ecd, enc, eval, btpEval)
			if err != nil {
				panic(fmt.Sprintf("failed to compute attention for head %d i: %v", j, err))
			}
			_ = ctAdd // ctAdd is not used, but computed for consistency
			ctAttentionHeads[j*stepSizeQ+k] = ctAttention
			// fmt.Printf("Layer %d, Head %d: Computed attention with %d ciphertexts\n", i, j, len(ctAttention.Ciphertexts))
		}
	}

	// 3.3) Combine the attention heads
	ctAttention := matrix.MergeCiphertextMatricesByHeads(ctAttentionHeads)

	return ctAttention, nil
	// return ctQKV, ctQKTInvSum, nil
}

// 由于llama的attention计算和bert不一样，单独写一个ComputeAttention函数
// 具体计算中ctQ的维度32，ctK的维度8，ctV的维度8
// 4个Q对应1个K和1个V
func ComputeGroupAttentionLlama(
	ctQ *he.CiphertextMatrices,
	ctK, ctV []*he.CiphertextMatrices,
	modelParams *configs.ModelParams,
	ckksParams *ckks.Parameters,
	ecd *ckks.Encoder,
	enc *rlwe.Encryptor,
	eval *ckks.Evaluator,
	btpEval *bootstrapping.Evaluator,
) (*he.CiphertextMatrices, *rlwe.Ciphertext, error) {
	//step.1 compute exp(x-modelParams.ExpSubValue)
	ctQKV, ctSum, err := matrix.GroupCiphertextMatricesComputeAttentionWithBSGSAndApproxMax(ctQ, ctK, ctV, modelParams, *ckksParams, eval)
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

func ComputeFFNLlama(
	ctInput *he.CiphertextMatrices,
	layerParams *utils.LlamaLayerParameters,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	eval *ckks.Evaluator,
) (*he.CiphertextMatrices, error) {
	// Q = ctInput × weightQ
	ctU, err := matrix.CiphertextMatricesMultiplyPlaintextMatrix(ctInput, modelParams, layerParams.MLPUpProjWeight.T(), ckksParams, eval)
	if err != nil {
		return nil, err
	}
	// K = ctInput × weightK
	ctV, err := matrix.CiphertextMatricesMultiplyPlaintextMatrix(ctInput, modelParams, layerParams.MLPGateProjWeight.T(), ckksParams, eval)
	if err != nil {
		return nil, err
	}

	ctSiLUV := silu.CiphertextMatricesSiLUChebyshev(ctV, &ckksParams, eval, modelParams.GeluMinValue, modelParams.GeluMaxValue, modelParams.GeluDegree)

	ctS, err := matrix.CiphertextMatricesMulCiphertextMatrices(ctU, ctSiLUV, eval)
	if err != nil {
		return nil, err
	}

	// V = ctInput × weightV
	ctFFNOutput, err := matrix.CiphertextMatricesMultiplyPlaintextMatrix(ctS, modelParams, layerParams.MLPDownProjWeight.T(), ckksParams, eval)
	if err != nil {
		return nil, err
	}
	return ctFFNOutput, nil
}
