package activation

import (
	"Arion/pkg/he"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/polynomial"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

// ApproximatePolynomialChebyshev 对 CiphertextMatrices 中每个密文，使用切比雪夫多项式在指定区间上进行多项式近似计算。
// 输入：
//   ctMats      - 需要近似的密文矩阵（*he.CiphertextMatrices）
//   poly        - 切比雪夫多项式（bignum.Polynomial），包含区间[a, b]和系数
//   ckksParams  - CKKS参数（*ckks.Parameters）
//   eval        - CKKS Evaluator
// 输出：
//   返回新的密文矩阵（*he.CiphertextMatrices），每个密文已做多项式近似
// 过程：
//   1. 首先将输入密文做区间线性变换，映射到切比雪夫区间
//   2. 然后用 polynomial.Evaluator 对每个密文应用切比雪夫多项式近似
//   3. 返回近似后的密文矩阵

func ApproximatePolynomialChebyshev(
	ctMats *he.CiphertextMatrices,
	poly bignum.Polynomial,
	ckksParams *ckks.Parameters,
	eval *ckks.Evaluator,
) *he.CiphertextMatrices {
	// First, we must operate the change of basis for the Chebyshev evaluation y = (2*x-a-b)/(b-a) = scalarmul * x + scalaradd
	scalarmul, scalaradd := poly.ChangeOfBasis()

	newCiphertexts := make([]*rlwe.Ciphertext, len(ctMats.Ciphertexts))
	for i := 0; i < len(ctMats.Ciphertexts); i++ {
		res, err := eval.MulNew(ctMats.Ciphertexts[i], scalarmul)
		if err != nil {
			panic(err)
		}
		if err = eval.Add(res, scalaradd, res); err != nil {
			panic(err)
		}
		if err = eval.Rescale(res, res); err != nil {
			panic(err)
		}
		polyEval := polynomial.NewEvaluator(*ckksParams, eval)
		if res, err = polyEval.Evaluate(res, poly, ckksParams.DefaultScale()); err != nil {
			panic(err)
		}
		newCiphertexts[i] = res
	}
	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ctMats.NumBatch,
		NumRow:      ctMats.NumRow,
		NumCol:      ctMats.NumCol,
	}
}

func ApproximatePolynomialChebyshevSingle(
	ct *rlwe.Ciphertext,
	poly bignum.Polynomial,
	ckksParams *ckks.Parameters,
	eval *ckks.Evaluator,
) *rlwe.Ciphertext {
	// First, we must operate the change of basis for the Chebyshev evaluation y = (2*x-a-b)/(b-a) = scalarmul * x + scalaradd
	scalarmul, scalaradd := poly.ChangeOfBasis()

	res, err := eval.MulNew(ct, scalarmul)
	if err != nil {
		panic(err)
	}
	if err = eval.Add(res, scalaradd, res); err != nil {
		panic(err)
	}
	if err = eval.Rescale(res, res); err != nil {
		panic(err)
	}
	polyEval := polynomial.NewEvaluator(*ckksParams, eval)

	if res, err = polyEval.Evaluate(res, poly, ckksParams.DefaultScale()); err != nil {
		panic(err)
	}

	return res
}
