package rmsnorm

import (
	"Arion/configs"
	"Arion/pkg/he"
	"Arion/pkg/math/activation"
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
	"gonum.org/v1/gonum/mat"
)

func CiphertextInverseSqrtChebyshev(
	ct *rlwe.Ciphertext,
	ckksParams *ckks.Parameters,
	eval *ckks.Evaluator,
	minValue float64,
	maxValue float64,
	degree int,
) *rlwe.Ciphertext {
	InvSqertX := func(x complex128) (y complex128) {
		return complex(1/math.Sqrt(real(x)), 0)
	}

	prec := ckksParams.EncodingPrecision()
	interval := bignum.Interval{
		Nodes: degree,
		A:     *bignum.NewFloat(minValue, prec),
		B:     *bignum.NewFloat(maxValue, prec),
	}
	poly := bignum.ChebyshevApproximation(InvSqertX, interval)

	// 这里 ApproximatePolynomialChebyshev 支持单个密文输入
	return activation.ApproximatePolynomialChebyshevSingle(ct, poly, ckksParams, eval)
}

func CiphertextInverseSqrtNewtonIteration(
	ctX *rlwe.Ciphertext,
	ctXInvInit *rlwe.Ciphertext,
	iter int,
	eval *ckks.Evaluator,
) *rlwe.Ciphertext {
	ctRes := ctXInvInit
	// x_{i+1}=3/2*x_i-c/2*x_i^3
	// x_i=ctXInvInit
	// c= ctX
	for i := 0; i < iter; i++ {
		// -1/2 * c
		ctTmp, err := eval.MulRelinNew(ctX, -0.5)
		if err != nil {
			panic(err)
		}
		if err := eval.Rescale(ctTmp, ctTmp); err != nil {
			panic(err)
		}

		// - 1/2 * c * x_i
		err = eval.MulRelin(ctTmp, ctRes, ctTmp)
		if err != nil {
			panic(err)
		}
		if err := eval.Rescale(ctTmp, ctTmp); err != nil {
			panic(err)
		}

		// 3/2 * x_i
		ctScaleXi, err := eval.MulRelinNew(ctRes, 1.5)
		if err != nil {
			panic(err)
		}
		if err := eval.Rescale(ctScaleXi, ctScaleXi); err != nil {
			panic(err)
		}

		// x_i^2
		ctSqure, err := eval.MulRelinNew(ctRes, ctRes)
		if err != nil {
			panic(err)
		}
		if err := eval.Rescale(ctSqure, ctSqure); err != nil {
			panic(err)
		}

		// 1/2 * c * x_i^3
		if err := eval.MulRelin(ctSqure, ctTmp, ctTmp); err != nil {
			panic(err)
		}
		if err := eval.Rescale(ctTmp, ctTmp); err != nil {
			panic(err)
		}

		// x_i*3/2 - 1/2 * c * x_i^3
		err = eval.Add(ctScaleXi, ctTmp, ctRes)
		if err != nil {
			panic(err)
		}
	}
	return ctRes
}

/*
 * InvertSqrtByChebyshevAndNewtonIter
 * Input:
 */
func InvertSqrtByChebyshevAndNewtonIter(
	ctX *rlwe.Ciphertext,
	ckksParams *ckks.Parameters,
	eval *ckks.Evaluator,
	minValue float64,
	maxValue float64,
	degree int,
	iter int,
) *rlwe.Ciphertext {
	// 用Chebyshev计算初值
	// fmt.Println("minValue:", minValue, "maxValue:", maxValue, "degree:", degree)
	ctXInvSqrtInit := CiphertextInverseSqrtChebyshev(ctX, ckksParams, eval, minValue, maxValue, degree)

	// 用牛顿迭代计算1/sqrt(x)
	ctRes := CiphertextInverseSqrtNewtonIteration(ctX, ctXInvSqrtInit, iter, eval)

	// Newton Iter
	return ctRes

}

/*
 * Modify Standard Layernorm
 *       r * sqrt(d) * d*x_i
 *   ——————————————————————————————————
 *      sqrt(SUM(d*x_i)^2)
 */

func RMSNormSelfAttentionInput(
	ctX *he.CiphertextMatrices,
	gammaMat mat.Matrix,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	ecd *ckks.Encoder,
	eval *ckks.Evaluator,
	// dec *rlwe.Decryptor,
) (*he.CiphertextMatrices, error) {
	numCt := len(ctX.Ciphertexts)
	var err error

	// ctBatch := modelParams.NumBatch
	// ctRealRow := modelParams.NumRealRow
	if numCt != modelParams.NumCol {
		return nil, fmt.Errorf("RNSNorm expects 4096 ciphertexts, got %d", numCt)
	}
	// 检查 gammaMat 和 betaMat 的列数是否等于 ctX.NumCol
	if gammaMat == nil {
		return nil, fmt.Errorf("RNSNorm: gammaMat 或 betaMat 为空")
	}
	gammaRows, gammaCols := gammaMat.Dims()
	if gammaRows != ctX.NumCol || gammaCols != 1 {
		return nil, fmt.Errorf("RNSNorm: gammaMat （%d）与输入密文矩阵列数（%d）不匹配", gammaRows, ctX.NumCol)
	}

	// 1. 计算SUM(d*x_i)^2
	ctSquareSum := ckks.NewCiphertext(ckksParams, ctX.Ciphertexts[0].Degree(), ctX.Ciphertexts[0].Level())
	for i := 0; i < numCt; i++ {
		if err = eval.MulRelinThenAdd(ctX.Ciphertexts[i], ctX.Ciphertexts[i], ctSquareSum); err != nil {
			panic(err)
		}
	}
	if err = eval.Rescale(ctSquareSum, ctSquareSum); err != nil {
		panic(err)
	}

	//** 加上d*eps=d*1e-5
	eval.Add(ctSquareSum, float64(numCt)*(1e-5), ctSquareSum)

	// 2. 计算 inv_sqrt_var = 1/sqrt(SUM(d*x_i - SUM(X)) / d^2)（用切比雪夫和牛顿迭代去逼近）
	invSqrtVar := InvertSqrtByChebyshevAndNewtonIter(ctSquareSum, &ckksParams, eval, modelParams.InvSqrtMinValue1, modelParams.InvSqrtMaxValue1, modelParams.InvSqrtDegree1, modelParams.InvSqrtIter1)

	// ptXSumSquare := he.DecryptCiphertext(invSqrtVar, ckksParams, dec, ecd)
	// fmt.Println("RMSNorm: invSqrtVar of ciphertexts:", ptXSumSquare[:8])

	// 3. 计算 r/srqt(d) *x
	newCiphertexts := make([]*rlwe.Ciphertext, numCt)
	for i := 0; i < numCt; i++ {
		//scale
		ctScale, err := eval.MulRelinNew(ctX.Ciphertexts[i], gammaMat.At(i, 0)*math.Sqrt(float64(numCt)))
		if err != nil {
			panic(err)
		}
		if err = eval.Rescale(ctScale, ctScale); err != nil {
			panic(err)
		}
		// *inv
		ctRes, err := eval.MulRelinNew(ctScale, invSqrtVar)
		if err != nil {
			panic(err)
		}
		if err = eval.Rescale(ctRes, ctRes); err != nil {
			panic(err)
		}
		newCiphertexts[i] = ctRes
	}

	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumRow:      ctX.NumRow,
		NumCol:      ctX.NumCol,
		NumBatch:    ctX.NumBatch,
	}, nil
}

func RMSNormFFNInput(
	ctX *he.CiphertextMatrices,
	gammaMat mat.Matrix,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	ecd *ckks.Encoder,
	eval *ckks.Evaluator,
	// dec *rlwe.Decryptor,
) (*he.CiphertextMatrices, error) {
	numCt := len(ctX.Ciphertexts)
	var err error

	// ctBatch := modelParams.NumBatch
	// ctRealRow := modelParams.NumRealRow
	if numCt != modelParams.NumCol {
		return nil, fmt.Errorf("RNSNorm expects 4096 ciphertexts, got %d", numCt)
	}
	// 检查 gammaMat 和 betaMat 的列数是否等于 ctX.NumCol
	if gammaMat == nil {
		return nil, fmt.Errorf("RNSNorm: gammaMat 或 betaMat 为空")
	}
	gammaRows, gammaCols := gammaMat.Dims()
	if gammaRows != ctX.NumCol || gammaCols != 1 {
		return nil, fmt.Errorf("RNSNorm: gammaMat （%d）与输入密文矩阵列数（%d）不匹配", gammaRows, ctX.NumCol)
	}

	// 1. 计算SUM(d*x_i)^2
	ctSquareSum := ckks.NewCiphertext(ckksParams, ctX.Ciphertexts[0].Degree(), ctX.Ciphertexts[0].Level())
	for i := 0; i < numCt; i++ {
		if err = eval.MulRelinThenAdd(ctX.Ciphertexts[i], ctX.Ciphertexts[i], ctSquareSum); err != nil {
			panic(err)
		}
	}
	if err = eval.Rescale(ctSquareSum, ctSquareSum); err != nil {
		panic(err)
	}

	//** 加上d*eps=d*1e-5
	eval.Add(ctSquareSum, float64(numCt)*(1e-5), ctSquareSum)

	// 2. 计算 inv_sqrt_var = 1/sqrt(SUM(d*x_i - SUM(X)) / d^2)（用切比雪夫和牛顿迭代去逼近）
	invSqrtVar := InvertSqrtByChebyshevAndNewtonIter(ctSquareSum, &ckksParams, eval, modelParams.InvSqrtMinValue2, modelParams.InvSqrtMaxValue2, modelParams.InvSqrtDegree2, modelParams.InvSqrtIter2)

	// ptXSumSquare := he.DecryptCiphertext(invSqrtVar, ckksParams, dec, ecd)
	// fmt.Println("RMSNorm: invSqrtVar of ciphertexts:", ptXSumSquare[:8])

	// 3. 计算 r/srqt(d) *x
	newCiphertexts := make([]*rlwe.Ciphertext, numCt)
	for i := 0; i < numCt; i++ {
		//scale
		ctScale, err := eval.MulRelinNew(ctX.Ciphertexts[i], gammaMat.At(i, 0)*math.Sqrt(float64(numCt)))
		if err != nil {
			panic(err)
		}
		if err = eval.Rescale(ctScale, ctScale); err != nil {
			panic(err)
		}
		// *inv
		ctRes, err := eval.MulRelinNew(ctScale, invSqrtVar)
		if err != nil {
			panic(err)
		}
		if err = eval.Rescale(ctRes, ctRes); err != nil {
			panic(err)
		}
		newCiphertexts[i] = ctRes
	}

	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumRow:      ctX.NumRow,
		NumCol:      ctX.NumCol,
		NumBatch:    ctX.NumBatch,
	}, nil
}
