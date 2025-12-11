package layernorm

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

// LayerNorm 对密文向量做 LayerNorm 操作
// x: []*rlwe.Ciphertext，长度为num_ct（如768）
// gamma, beta: 明文参数，长度同x
// biasVec: 掩码，长度等于slots
// 返回归一化后的密文向量
func LayerNormStandard(
	ctX *he.CiphertextMatrices,
	gammaMat, betaMat mat.Matrix,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	ecd *ckks.Encoder,
	eval *ckks.Evaluator,
	minValue float64,
	maxValue float64,
	degree int,
	iter int,
) (*he.CiphertextMatrices, error) {
	numCt := len(ctX.Ciphertexts)
	var err error

	if numCt != modelParams.NumCol {
		return nil, fmt.Errorf("LayerNorm expects 768 ciphertexts, got %d", numCt)
	}
	// 检查 gammaMat 和 betaMat 的列数是否等于 ctX.NumCol
	if gammaMat == nil || betaMat == nil {
		return nil, fmt.Errorf("LayerNorm: gammaMat 或 betaMat 为空")
	}
	gammaRows, gammaCols := gammaMat.Dims()
	betaRows, betaCols := betaMat.Dims()
	if gammaRows != ctX.NumCol || betaRows != ctX.NumCol || gammaCols != 1 || betaCols != 1 {
		return nil, fmt.Errorf("LayerNorm: gammaMat 或 betaMat 的行数（%d, %d）与输入密文矩阵列数（%d）不匹配", gammaRows, betaRows, ctX.NumCol)
	}

	// 1.计算均值和方差
	ctAvg, ctVar, err := CiphertextMatricesReturnAvgAndVarNorm(ctX, modelParams, ckksParams, eval)
	if err != nil {
		return nil, fmt.Errorf("LayerNorm: failed to compute average and variance: %v", err)
	}
	_ = ctAvg

	// 2. 计算 inv_sqrt_var = 1/sqrt(var)(先使用切比雪夫寻找初值，然后使用牛顿迭代计算结果)
	invSqrtVar := InvertSqrtByChebyshevAndNewtonIter(ctVar, &ckksParams, eval, minValue, maxValue, degree, iter)

	// 3. 归一化输出
	newCiphertexts := make([]*rlwe.Ciphertext, numCt)
	for i := 0; i < numCt; i++ {
		ctSub, err := eval.SubNew(ctX.Ciphertexts[i], ctAvg)
		if err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to subtract average: %v", err))
		}
		// 乘以Gamma
		ctMulGamma, err := eval.MulRelinNew(ctSub, gammaMat.At(i, 0))
		if err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to multiply by gamma: %v", err))
		}
		if err := eval.Rescale(ctMulGamma, ctMulGamma); err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to rescale after multiplication by gamma: %v", err))
		}

		// 乘以1/sqrt(var)
		ctMulInvSqrtVar, err := eval.MulRelinNew(ctMulGamma, invSqrtVar)
		if err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to multiply by inverse sqrt variance: %v", err))
		}
		if err := eval.Rescale(ctMulInvSqrtVar, ctMulInvSqrtVar); err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to rescale after multiplication by inverse sqrt variance: %v", err))
		}
		// 加上Beta
		err = eval.Add(ctMulInvSqrtVar, betaMat.At(i, 0), ctMulInvSqrtVar)
		if err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to add beta: %v", err))
		}
		newCiphertexts[i] = ctMulInvSqrtVar
	}
	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumRow:      ctX.NumRow,
		NumCol:      ctX.NumCol,
		NumBatch:    ctX.NumBatch,
	}, nil
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

func InvertSqrtByChebyshevAndNewtonIterTest(
	ctX *rlwe.Ciphertext,
	ckksParams *ckks.Parameters,
	eval *ckks.Evaluator,
	minValue float64,
	maxValue float64,
	degree int,
	iter int,
	ecd *ckks.Encoder,
	dec *rlwe.Decryptor,
) *rlwe.Ciphertext {
	// 用Chebyshev计算初值
	fmt.Println("minValue:", minValue, "maxValue:", maxValue, "degree:", degree)
	ctXInvSqrtInit := CiphertextInverseSqrtChebyshev(ctX, ckksParams, eval, minValue, maxValue, degree)

	valueInv := he.DecryptCiphertext(ctXInvSqrtInit, *ckksParams, dec, ecd)
	fmt.Println("LayerNormOutput: invSqrtVarinit of ciphertexts:", valueInv[:20])
	// 用牛顿迭代计算1/sqrt(x)
	ctRes := CiphertextInverseSqrtNewtonIteration(ctX, ctXInvSqrtInit, iter, eval)

	// Newton Iter
	return ctRes

}

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

// func CiphertextInverseSqrtNewtonIteration(
// 	ctX *rlwe.Ciphertext,
// 	ctXInvInit *rlwe.Ciphertext,
// 	iter int,
// 	eval *ckks.Evaluator,
// ) *rlwe.Ciphertext {
// 	ctRes := ctXInvInit
// 	// x_{i+1}=3/2*x_i-c/2*x_i^3
// 	// x_i=ctXInvInit
// 	// c= ctX
// 	for i := 0; i < iter; i++ {
// 		// -1/2 * c
// 		ctTmp, err := eval.MulRelinNew(ctX, -0.5)
// 		if err != nil {
// 			panic(err)
// 		}
// 		if err := eval.Rescale(ctTmp, ctTmp); err != nil {
// 			panic(err)
// 		}

// 		// x_i^2
// 		ctSqure, err := eval.MulRelinNew(ctRes, ctRes)
// 		if err != nil {
// 			panic(err)
// 		}
// 		if err := eval.Rescale(ctSqure, ctSqure); err != nil {
// 			panic(err)
// 		}

// 		// - 1/2 * c * x_i^2
// 		err = eval.MulRelin(ctTmp, ctSqure, ctTmp)
// 		if err != nil {
// 			panic(err)
// 		}
// 		if err := eval.Rescale(ctTmp, ctTmp); err != nil {
// 			panic(err)
// 		}

// 		// 3/2 - 1/2 * c * x_i^2
// 		err = eval.Add(ctTmp, 1.5, ctTmp)
// 		if err != nil {
// 			panic(err)
// 		}

// 		// x_i(3/2 - 1/2 * c * x_i^2)
// 		if err := eval.MulRelin(ctRes, ctTmp, ctRes); err != nil {
// 			panic(err)
// 		}
// 		if err := eval.Rescale(ctRes, ctRes); err != nil {
// 			panic(err)
// 		}
// 	}
// 	return ctRes
// }

/*
 * Modify Standard Layernorm
 *    r                  d*x_i -SUM(X)
 *  ——————  *  ——————————————————————————————————   + b
 *  sqrt(d)     sqrt(SUM(d*x_i - SUM(X))^2 / d^2)
 */

func LayerNormSelfAttentionOutput(
	ctX *he.CiphertextMatrices,
	gammaMat, betaMat mat.Matrix,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	ecd *ckks.Encoder,
	eval *ckks.Evaluator,
	// dec *rlwe.Decryptor,
) (*he.CiphertextMatrices, error) {
	numCt := len(ctX.Ciphertexts)
	var err error

	ctBatch := modelParams.NumBatch
	ctRealRow := modelParams.NumRealRow
	if numCt != modelParams.NumCol {
		return nil, fmt.Errorf("LayerNorm expects 768 ciphertexts, got %d", numCt)
	}
	// 检查 gammaMat 和 betaMat 的列数是否等于 ctX.NumCol
	if gammaMat == nil || betaMat == nil {
		return nil, fmt.Errorf("LayerNorm: gammaMat 或 betaMat 为空")
	}
	gammaRows, gammaCols := gammaMat.Dims()
	betaRows, betaCols := betaMat.Dims()
	if gammaRows != ctX.NumCol || betaRows != ctX.NumCol || gammaCols != 1 || betaCols != 1 {
		return nil, fmt.Errorf("LayerNorm: gammaMat 或 betaMat 的行数（%d, %d）与输入密文矩阵列数（%d）不匹配", gammaRows, betaRows, ctX.NumCol)
	}

	// 1. 计算 放缩所有x的值 和 sum_x = x0 + x1 + ... + xN
	ctSum := ckks.NewCiphertext(ckksParams, ctX.Ciphertexts[0].Degree(), ctX.Ciphertexts[0].Level())
	for i := 0; i < numCt; i++ {
		// 求和
		eval.Add(ctSum, ctX.Ciphertexts[i], ctSum)
		// 放缩
		if err := eval.Mul(ctX.Ciphertexts[i], numCt, ctX.Ciphertexts[i]); err != nil {
			panic(err)
		}
	}

	// 2. 计算SUM(d*x_i - SUM(X))^2 / d^2
	ctXSubSumSquare := make([]*rlwe.Ciphertext, numCt) // 存储d*x_i - SUM(X),便于后续计算
	ctSumSquare := ckks.NewCiphertext(ckksParams, ctX.Ciphertexts[0].Degree(), ctX.Ciphertexts[0].Level())
	for i := 0; i < numCt; i++ {
		ctTmp, err := eval.SubNew(ctX.Ciphertexts[i], ctSum)
		if err != nil {
			panic(err)
		}
		if err = eval.MulRelinThenAdd(ctTmp, ctTmp, ctSumSquare); err != nil {
			panic(err)
		}
		ctXSubSumSquare[i] = ctTmp
	}
	if err = eval.Rescale(ctSumSquare, ctSumSquare); err != nil {
		panic(err)
	}
	if err = eval.MulRelin(ctSumSquare, 1/float64(numCt*numCt), ctSumSquare); err != nil {
		panic(err)
	}
	if err = eval.Rescale(ctSumSquare, ctSumSquare); err != nil {
		panic(err)
	}

	// 3. 计算 inv_sqrt_var = 1/sqrt(SUM(d*x_i - SUM(X)) / d^2)（用切比雪夫和牛顿迭代去逼近）
	invSqrtVar := InvertSqrtByChebyshevAndNewtonIter(ctSumSquare, &ckksParams, eval, modelParams.InvSqrtMinValue1*float64(numCt), modelParams.InvSqrtMaxValue1*float64(numCt), modelParams.InvSqrtDegree1, modelParams.InvSqrtIter1)

	// 4. 归一化输出
	newCiphertexts := make([]*rlwe.Ciphertext, numCt) // 存储d*x_i - SUM(X),便于后续计算
	for i := 0; i < numCt; i++ {

		baseVal := float64(gammaMat.At(i, 0)) / math.Sqrt(float64(numCt))
		plainVector := make([]float64, ctBatch*ctRealRow)
		for j := 0; j < ctBatch*ctRealRow; j++ {
			plainVector[j] = baseVal
		}
		// 乘以Gamma/sqrt(d)
		ctMulGamma, err := eval.MulRelinNew(ctXSubSumSquare[i], plainVector)
		if err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to multiply by gamma: %v", err))
		}
		if err := eval.Rescale(ctMulGamma, ctMulGamma); err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to rescale after multiplication by gamma: %v", err))
		}

		// 乘以1/sqrt(var)
		ctMulInvSqrtVar, err := eval.MulRelinNew(ctMulGamma, invSqrtVar)
		if err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to multiply by inverse sqrt variance: %v", err))
		}
		if err := eval.Rescale(ctMulInvSqrtVar, ctMulInvSqrtVar); err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to rescale after multiplication by inverse sqrt variance: %v", err))
		}
		// 加上Beta
		biasVal := betaMat.At(i, 0)
		biasVec := make([]float64, ctBatch*ctRealRow)
		for j := 0; j < ctBatch*ctRealRow; j++ {
			biasVec[j] = biasVal
		}

		err = eval.Add(ctMulInvSqrtVar, biasVec, ctMulInvSqrtVar)
		if err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to add beta: %v", err))
		}
		newCiphertexts[i] = ctMulInvSqrtVar
	}
	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumRow:      ctX.NumRow,
		NumCol:      ctX.NumCol,
		NumBatch:    ctX.NumBatch,
	}, nil
}

func LayerNormSelfAttentionOutputTest(
	ctX *he.CiphertextMatrices,
	gammaMat, betaMat mat.Matrix,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	ecd *ckks.Encoder,
	eval *ckks.Evaluator,
	dec *rlwe.Decryptor,
) (*he.CiphertextMatrices, error) {
	numCt := len(ctX.Ciphertexts)
	var err error

	ctBatch := modelParams.NumBatch
	ctRealRow := modelParams.NumRealRow
	if numCt != modelParams.NumCol {
		return nil, fmt.Errorf("LayerNorm expects 768 ciphertexts, got %d", numCt)
	}
	// 检查 gammaMat 和 betaMat 的列数是否等于 ctX.NumCol
	if gammaMat == nil || betaMat == nil {
		return nil, fmt.Errorf("LayerNorm: gammaMat 或 betaMat 为空")
	}
	gammaRows, gammaCols := gammaMat.Dims()
	betaRows, betaCols := betaMat.Dims()
	if gammaRows != ctX.NumCol || betaRows != ctX.NumCol || gammaCols != 1 || betaCols != 1 {
		return nil, fmt.Errorf("LayerNorm: gammaMat 或 betaMat 的行数（%d, %d）与输入密文矩阵列数（%d）不匹配", gammaRows, betaRows, ctX.NumCol)
	}

	// 1. 计算 放缩所有x的值 和 sum_x = x0 + x1 + ... + xN
	ctSum := ckks.NewCiphertext(ckksParams, ctX.Ciphertexts[0].Degree(), ctX.Ciphertexts[0].Level())
	for i := 0; i < numCt; i++ {
		// 求和
		eval.Add(ctSum, ctX.Ciphertexts[i], ctSum)
		// 放缩
		if err := eval.Mul(ctX.Ciphertexts[i], numCt, ctX.Ciphertexts[i]); err != nil {
			panic(err)
		}
	}

	ptSum := he.DecryptCiphertext(ctSum, ckksParams, dec, ecd)
	fmt.Println("LayerNorm: sum of ciphertexts:", ptSum[:8])

	ptScale := he.DecryptCiphertext(ctX.Ciphertexts[0], ckksParams, dec, ecd)
	fmt.Println("LayerNorm: scale of ciphertexts:", ptScale[:8])

	// 2. 计算SUM(d*x_i - SUM(X)) / d^2
	ctXSubSumSquare := make([]*rlwe.Ciphertext, numCt) // 存储d*x_i - SUM(X),便于后续计算
	ctSumSquare := ckks.NewCiphertext(ckksParams, ctX.Ciphertexts[0].Degree(), ctX.Ciphertexts[0].Level())
	for i := 0; i < numCt; i++ {
		ctTmp, err := eval.SubNew(ctX.Ciphertexts[i], ctSum)
		if err != nil {
			panic(err)
		}
		if err = eval.MulRelinThenAdd(ctTmp, ctTmp, ctSumSquare); err != nil {
			panic(err)
		}
		ctXSubSumSquare[i] = ctTmp
	}
	if err = eval.Rescale(ctSumSquare, ctSumSquare); err != nil {
		panic(err)
	}
	if err = eval.MulRelin(ctSumSquare, 1/float64(numCt*numCt), ctSumSquare); err != nil {
		panic(err)
	}
	if err = eval.Rescale(ctSumSquare, ctSumSquare); err != nil {
		panic(err)
	}

	ptXSubSumSquare := he.DecryptCiphertext(ctSumSquare, ckksParams, dec, ecd)
	fmt.Println("LayerNorm: XSubSumSquar of ciphertexts:", ptXSubSumSquare[:8])

	// 3. 计算 inv_sqrt_var = 1/sqrt(SUM(d*x_i - SUM(X)) / d^2)（用切比雪夫和牛顿迭代去逼近）
	invSqrtVar := InvertSqrtByChebyshevAndNewtonIter(ctSumSquare, &ckksParams, eval, modelParams.InvSqrtMinValue1*float64(numCt), modelParams.InvSqrtMaxValue1*float64(numCt), modelParams.InvSqrtDegree1, modelParams.InvSqrtIter1)
	ptSqrtVar := he.DecryptCiphertext(invSqrtVar, ckksParams, dec, ecd)
	fmt.Println("LayerNorm: invSqrtVar of ciphertexts:", ptSqrtVar[:8])

	// 3. 归一化输出
	newCiphertexts := make([]*rlwe.Ciphertext, numCt) // 存储d*x_i - SUM(X),便于后续计算
	for i := 0; i < numCt; i++ {

		baseVal := float64(gammaMat.At(i, 0)) / math.Sqrt(float64(numCt))
		plainVector := make([]float64, ctBatch*ctRealRow)
		for j := 0; j < ctBatch*ctRealRow; j++ {
			plainVector[j] = baseVal
		}
		fmt.Println("LayerNorm: baseVal for gamma:", baseVal)
		// 乘以Gamma/sqrt(d)
		ctMulGamma, err := eval.MulRelinNew(ctXSubSumSquare[i], plainVector)
		if err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to multiply by gamma: %v", err))
		}
		if err := eval.Rescale(ctMulGamma, ctMulGamma); err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to rescale after multiplication by gamma: %v", err))
		}

		ptMulGamma := he.DecryptCiphertext(ctMulGamma, ckksParams, dec, ecd)
		fmt.Println("LayerNorm: MulGamma of ciphertexts:", ptMulGamma[:8])

		// 乘以1/sqrt(var)
		ctMulInvSqrtVar, err := eval.MulRelinNew(ctMulGamma, invSqrtVar)
		if err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to multiply by inverse sqrt variance: %v", err))
		}
		if err := eval.Rescale(ctMulInvSqrtVar, ctMulInvSqrtVar); err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to rescale after multiplication by inverse sqrt variance: %v", err))
		}

		// 加上Beta
		biasVal := betaMat.At(i, 0)
		biasVec := make([]float64, ctBatch*ctRealRow)
		for j := 0; j < ctBatch*ctRealRow; j++ {
			biasVec[j] = biasVal
		}

		err = eval.Add(ctMulInvSqrtVar, biasVec, ctMulInvSqrtVar)
		if err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to add beta: %v", err))
		}
		ptMulInvSqrtVar := he.DecryptCiphertext(ctMulInvSqrtVar, ckksParams, dec, ecd)
		fmt.Println("LayerNorm: MulInvSqrtVar of ciphertexts:", ptMulInvSqrtVar[:8])
		newCiphertexts[i] = ctMulInvSqrtVar
	}
	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumRow:      ctX.NumRow,
		NumCol:      ctX.NumCol,
		NumBatch:    ctX.NumBatch,
	}, nil
}

/*
 * Modify Standard Layernorm
 *    r                  d*x_i -SUM(X)
 *  ——————  *  ——————————————————————————————————   + b
 *    d          sqrt(SUM(d*x_i - SUM(X)) / d^3)
 */

func LayerNormOutput(
	ctX *he.CiphertextMatrices,
	gammaMat, betaMat mat.Matrix,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	ecd *ckks.Encoder,
	eval *ckks.Evaluator,
) (*he.CiphertextMatrices, error) {
	numCt := len(ctX.Ciphertexts)
	var err error

	ctBatch := modelParams.NumBatch
	ctRealRow := modelParams.NumRealRow

	if numCt != modelParams.NumCol {
		return nil, fmt.Errorf("LayerNorm expects 768 ciphertexts, got %d", numCt)
	}
	// 检查 gammaMat 和 betaMat 的列数是否等于 ctX.NumCol
	if gammaMat == nil || betaMat == nil {
		return nil, fmt.Errorf("LayerNorm: gammaMat 或 betaMat 为空")
	}
	gammaRows, gammaCols := gammaMat.Dims()
	betaRows, betaCols := betaMat.Dims()
	if gammaRows != ctX.NumCol || betaRows != ctX.NumCol || gammaCols != 1 || betaCols != 1 {
		return nil, fmt.Errorf("LayerNorm: gammaMat 或 betaMat 的行数（%d, %d）与输入密文矩阵列数（%d）不匹配", gammaRows, betaRows, ctX.NumCol)
	}

	// 1. 计算 放缩所有x的值 和 sum_x = x0 + x1 + ... + xN
	ctSum := ckks.NewCiphertext(ckksParams, ctX.Ciphertexts[0].Degree(), ctX.Ciphertexts[0].Level())
	for i := 0; i < numCt; i++ {
		// 求和
		eval.Add(ctSum, ctX.Ciphertexts[i], ctSum)
		// 放缩
		if err := eval.Mul(ctX.Ciphertexts[i], numCt, ctX.Ciphertexts[i]); err != nil {
			panic(err)
		}
	}

	// fmt.Println("LayerNormOutput: ctSum Level:", ctSum.Level())

	// 2. 计算SUM(d*x_i - SUM(X)) / d^3
	ctXSubSumSquare := make([]*rlwe.Ciphertext, numCt) // 存储d*x_i - SUM(X),便于后续计算
	ctSumSquare := ckks.NewCiphertext(ckksParams, ctX.Ciphertexts[0].Degree(), ctX.Ciphertexts[0].Level())
	for i := 0; i < numCt; i++ {
		ctTmp, err := eval.SubNew(ctX.Ciphertexts[i], ctSum)
		if err != nil {
			panic(err)
		}
		if err = eval.MulRelinThenAdd(ctTmp, ctTmp, ctSumSquare); err != nil {
			panic(err)
		}
		ctXSubSumSquare[i] = ctTmp
	}
	if err = eval.Rescale(ctSumSquare, ctSumSquare); err != nil {
		panic(err)
	}
	if err = eval.MulRelin(ctSumSquare, 1/float64(numCt*numCt*numCt), ctSumSquare); err != nil {
		panic(err)
	}
	if err = eval.Rescale(ctSumSquare, ctSumSquare); err != nil {
		panic(err)
	}

	// fmt.Println("LayerNormOutput: ctSumSquare Level:", ctSumSquare.Level())

	// 3. 计算 inv_sqrt_var = 1/sqrt(SUM(d*x_i - SUM(X)) / d^3)（用切比雪夫和牛顿迭代去逼近）
	invSqrtVar := InvertSqrtByChebyshevAndNewtonIter(ctSumSquare, &ckksParams, eval, modelParams.InvSqrtMinValue2, modelParams.InvSqrtMaxValue2, modelParams.InvSqrtDegree2, modelParams.InvSqrtIter2)

	// fmt.Println("LayerNormOutput: invSqrtVar Level:", invSqrtVar.Level())
	// 3. 归一化输出
	newCiphertexts := make([]*rlwe.Ciphertext, numCt) // 存储d*x_i - SUM(X),便于后续计算
	for i := 0; i < numCt; i++ {

		baseVal := float64(gammaMat.At(i, 0)) / float64(numCt)
		plainVector := make([]float64, ctBatch*ctRealRow)
		for j := 0; j < ctBatch*ctRealRow; j++ {
			plainVector[j] = baseVal
		}

		// 乘以Gamma/sqrt(d)
		ctMulGamma, err := eval.MulRelinNew(ctXSubSumSquare[i], plainVector)
		if err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to multiply by gamma: %v", err))
		}
		if err := eval.Rescale(ctMulGamma, ctMulGamma); err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to rescale after multiplication by gamma: %v", err))
		}

		// 乘以1/sqrt(var)
		ctMulInvSqrtVar, err := eval.MulRelinNew(ctMulGamma, invSqrtVar)
		if err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to multiply by inverse sqrt variance: %v", err))
		}
		if err := eval.Rescale(ctMulInvSqrtVar, ctMulInvSqrtVar); err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to rescale after multiplication by inverse sqrt variance: %v", err))
		}

		biasVal := betaMat.At(i, 0)
		biasVec := make([]float64, ctBatch*ctRealRow)
		for j := 0; j < ctBatch*ctRealRow; j++ {
			biasVec[j] = biasVal
		}
		// 加上Beta
		err = eval.Add(ctMulInvSqrtVar, biasVec, ctMulInvSqrtVar)
		if err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to add beta: %v", err))
		}
		newCiphertexts[i] = ctMulInvSqrtVar
	}
	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumRow:      ctX.NumRow,
		NumCol:      ctX.NumCol,
		NumBatch:    ctX.NumBatch,
	}, nil
}

func LayerNormOutputTest(
	ctX *he.CiphertextMatrices,
	gammaMat, betaMat mat.Matrix,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	ecd *ckks.Encoder,
	eval *ckks.Evaluator,
	dec *rlwe.Decryptor,
) (*he.CiphertextMatrices, error) {
	numCt := len(ctX.Ciphertexts)
	var err error

	ctBatch := modelParams.NumBatch
	ctRealRow := modelParams.NumRealRow

	if numCt != modelParams.NumCol {
		return nil, fmt.Errorf("LayerNorm expects 768 ciphertexts, got %d", numCt)
	}
	// 检查 gammaMat 和 betaMat 的列数是否等于 ctX.NumCol
	if gammaMat == nil || betaMat == nil {
		return nil, fmt.Errorf("LayerNorm: gammaMat 或 betaMat 为空")
	}
	gammaRows, gammaCols := gammaMat.Dims()
	betaRows, betaCols := betaMat.Dims()
	if gammaRows != ctX.NumCol || betaRows != ctX.NumCol || gammaCols != 1 || betaCols != 1 {
		return nil, fmt.Errorf("LayerNorm: gammaMat 或 betaMat 的行数（%d, %d）与输入密文矩阵列数（%d）不匹配", gammaRows, betaRows, ctX.NumCol)
	}

	// 1. 计算 放缩所有x的值 和 sum_x = x0 + x1 + ... + xN
	ctSum := ckks.NewCiphertext(ckksParams, ctX.Ciphertexts[0].Degree(), ctX.Ciphertexts[0].Level())
	for i := 0; i < numCt; i++ {
		ctOp := ctX.Ciphertexts[i]
		// 求和
		eval.Add(ctSum, ctOp, ctSum)
		// 放缩
		if err := eval.Mul(ctOp, numCt, ctOp); err != nil {
			panic(err)
		}
		ctX.Ciphertexts[i] = ctOp
	}
	valueSum := he.DecryptCiphertext(ctSum, ckksParams, dec, ecd)
	fmt.Println("LayerNormOutput: valueSum of ciphertexts:", valueSum[:30])

	// fmt.Println("LayerNormOutput: ctSum Level:", ctSum.Level())

	// 2. 计算SUM(d*x_i - SUM(X)) / d^3
	ctXSubSumSquare := make([]*rlwe.Ciphertext, numCt) // 存储d*x_i - SUM(X),便于后续计算
	ctSumSquare := ckks.NewCiphertext(ckksParams, ctX.Ciphertexts[0].Degree(), ctX.Ciphertexts[0].Level())
	for i := 0; i < numCt; i++ {
		ctTmp, err := eval.SubNew(ctX.Ciphertexts[i], ctSum)
		if err != nil {
			panic(err)
		}

		ctTmpScale, err := eval.MulRelinNew(ctTmp, math.Sqrt(1/float64(numCt*numCt*numCt)))
		if err != nil {
			panic(err)
		}
		if err = eval.Rescale(ctTmpScale, ctTmpScale); err != nil {
			panic(err)
		}

		if err = eval.MulRelinThenAdd(ctTmpScale, ctTmpScale, ctSumSquare); err != nil {
			panic(err)
		}

		ctXSubSumSquare[i] = ctTmp
	}
	if err = eval.Rescale(ctSumSquare, ctSumSquare); err != nil {
		panic(err)
	}
	// if err = eval.MulRelin(ctSumSquare, 1/float64(numCt*numCt*numCt), ctSumSquare); err != nil {
	// 	panic(err)
	// }
	// if err = eval.Rescale(ctSumSquare, ctSumSquare); err != nil {
	// 	panic(err)
	// }

	valueSigma := he.DecryptCiphertext(ctSumSquare, ckksParams, dec, ecd)
	fmt.Println("LayerNormOutput: valueSigma of ciphertexts:", valueSigma[:30])

	// fmt.Println("LayerNormOutput: ctSumSquare Level:", ctSumSquare.Level())

	// 3. 计算 inv_sqrt_var = 1/sqrt(SUM(d*x_i - SUM(X)) / d^3)（用切比雪夫和牛顿迭代去逼近）
	invSqrtVar := InvertSqrtByChebyshevAndNewtonIterTest(ctSumSquare, &ckksParams, eval, modelParams.InvSqrtMinValue2, modelParams.InvSqrtMaxValue2, modelParams.InvSqrtDegree2, modelParams.InvSqrtIter2, ecd, dec)

	valueInv := he.DecryptCiphertext(invSqrtVar, ckksParams, dec, ecd)
	fmt.Println("LayerNormOutput: invSqrtVar of ciphertexts:", valueInv[:20])

	// fmt.Println("LayerNormOutput: invSqrtVar Level:", invSqrtVar.Level())
	// 3. 归一化输出
	newCiphertexts := make([]*rlwe.Ciphertext, numCt) // 存储d*x_i - SUM(X),便于后续计算
	for i := 0; i < numCt; i++ {

		baseVal := float64(gammaMat.At(i, 0)) / float64(numCt)
		plainVector := make([]float64, ctBatch*ctRealRow)
		for j := 0; j < ctBatch*ctRealRow; j++ {
			plainVector[j] = baseVal
		}

		// 乘以Gamma/sqrt(d)
		ctMulGamma, err := eval.MulRelinNew(ctXSubSumSquare[i], plainVector)
		if err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to multiply by gamma: %v", err))
		}
		if err := eval.Rescale(ctMulGamma, ctMulGamma); err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to rescale after multiplication by gamma: %v", err))
		}

		// 乘以1/sqrt(var)
		ctMulInvSqrtVar, err := eval.MulRelinNew(ctMulGamma, invSqrtVar)
		if err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to multiply by inverse sqrt variance: %v", err))
		}
		if err := eval.Rescale(ctMulInvSqrtVar, ctMulInvSqrtVar); err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to rescale after multiplication by inverse sqrt variance: %v", err))
		}

		biasVal := betaMat.At(i, 0)
		biasVec := make([]float64, ctBatch*ctRealRow)
		for j := 0; j < ctBatch*ctRealRow; j++ {
			biasVec[j] = biasVal
		}
		// 加上Beta
		err = eval.Add(ctMulInvSqrtVar, biasVec, ctMulInvSqrtVar)
		if err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to add beta: %v", err))
		}
		newCiphertexts[i] = ctMulInvSqrtVar
	}
	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumRow:      ctX.NumRow,
		NumCol:      ctX.NumCol,
		NumBatch:    ctX.NumBatch,
	}, nil
}

func CiphertextMatricesReturnAvgAndVarNorm(
	ctX *he.CiphertextMatrices,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	eval *ckks.Evaluator,
) (*rlwe.Ciphertext, *rlwe.Ciphertext, error) {
	numCt := len(ctX.Ciphertexts)
	// 1. 计算 和 sum_x = x0 + x1 + ... + xN
	ctSum := ckks.NewCiphertext(ckksParams, ctX.Ciphertexts[0].Degree(), ctX.Ciphertexts[0].Level())
	for i := 1; i < numCt; i++ {
		eval.Add(ctSum, ctX.Ciphertexts[i], ctSum)
	}
	// 2. 计算 均值 (n=768, 只在有数值的位置进行计算)
	ctAvg, err := eval.MulRelinNew(ctSum, 1.0/float64(numCt))
	if err != nil {
		panic(fmt.Sprintf("LayerNorm: failed to compute average: %v", err))
	}
	if err := eval.Rescale(ctAvg, ctAvg); err != nil {
		panic(fmt.Sprintf("LayerNorm: failed to rescale average: %v", err))
	}

	// 3. 计算 方差 (n=768, 只在有数值的位置进行计算)
	ctSqure := ckks.NewCiphertext(ckksParams, ctAvg.Degree(), ctAvg.Level())
	for i := 0; i < numCt; i++ {
		// 减去均值
		ctSub, err := eval.SubNew(ctX.Ciphertexts[i], ctAvg)
		if err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to compute variance: %v", err))
		}
		// 平方
		err = eval.MulRelinThenAdd(ctSub, ctSub, ctSqure)
		if err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to compute square: %v", err))
		}
	}
	if err := eval.Rescale(ctSqure, ctSqure); err != nil {
		panic(fmt.Sprintf("LayerNorm: failed to rescale variance: %v", err))
	}

	// 4. 计算 方差的均值
	ctVar, err := eval.MulRelinNew(ctSqure, float64(1/float64(numCt)))
	if err != nil {
		panic(err)
	}
	if err := eval.Rescale(ctVar, ctVar); err != nil {
		panic(fmt.Sprintf("LayerNorm: failed to rescale variance: %v", err))
	}
	return ctAvg, ctVar, nil
}

// InvertSqrt 近似计算 1/sqrt(x) 的密文
func InvertSqrt(
	x *rlwe.Ciphertext,
	dNewton, dGold int,
	ckksParams ckks.Parameters,
	eval *ckks.Evaluator,
) *rlwe.Ciphertext {

	// 1. 计算高斯变换
	res := initGuess(x, eval)

	// 2. 牛顿迭代求解
	y := newtonIter(x, res, dNewton, ckksParams, eval)

	// 3. Goldschmidt迭代求解
	sqrtInv := goldSchmidtIter(x, y, dGold, ckksParams, eval)

	return sqrtInv
}

func initGuess(
	x *rlwe.Ciphertext,
	eval *ckks.Evaluator,
) *rlwe.Ciphertext {
	// 计算x*-1.29054537e-04+1.29054537e-01
	ctGuess, err := eval.MulRelinNew(x, -1.29054537e-04)
	if err != nil {
		panic(fmt.Sprintf("LayerNorm: failed to compute initial guess: %v", err))
	}
	if err := eval.Rescale(ctGuess, ctGuess); err != nil {
		panic(fmt.Sprintf("LayerNorm: failed to rescale initial guess: %v", err))
	}
	err = eval.Add(ctGuess, 1.29054537e-01, ctGuess)
	if err != nil {
		panic(fmt.Sprintf("LayerNorm: failed to add initial guess: %v", err))
	}
	return ctGuess
}

// newtonIter 对密文res进行iter次牛顿迭代，近似1/sqrt(x)
// x: 原始密文
// res: 初始猜测密文
// iter: 迭代次数
// ckksParams: CKKS参数
// eval: Evaluator
// relinKeys: 重线性化密钥
func newtonIter(
	x, res *rlwe.Ciphertext,
	iter int,
	ckksParams ckks.Parameters,
	eval *ckks.Evaluator,
) *rlwe.Ciphertext {

	for i := 0; i < iter; i++ {

		// res_sq = res^2
		resSq, err := eval.MulRelinNew(res, res)
		if err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to compute square: %v", err))
		}
		if err := eval.Rescale(resSq, resSq); err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to rescale square: %v", err))
		}

		// res_x = x * (-0.5)
		resX, err := eval.MulRelinNew(x, -0.5)
		if err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to compute res_x: %v", err))
		}
		if err := eval.Rescale(resX, resX); err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to rescale res_x: %v", err))
		}

		// res_x = res_x * res (-0.5*x*b)
		if err = eval.MulRelin(resX, res, resX); err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to multiply res_x with res: %v", err))
		}
		if err = eval.Rescale(resX, resX); err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to rescale res_x: %v", err))
		}

		// res_x = res_x * res_sq (-0.5*b*x^3)
		if err = eval.MulRelin(resX, resSq, resX); err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to multiply res_x with res_sq: %v", err))
		}
		if err = eval.Rescale(resX, resX); err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to rescale res_x: %v", err))
		}

		// res = res * 1.5 (1.5*x)
		err = eval.MulRelin(res, 1.5, res)
		if err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to compute res1: %v", err))
		}
		if err := eval.Rescale(res, res); err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to rescale res1: %v", err))
		}

		//-0.5*b*x^3 + 1.5*x
		res.Scale = ckksParams.DefaultScale()
		resX.Scale = ckksParams.DefaultScale()
		if err = eval.Add(res, resX, res); err != nil {
			panic(fmt.Sprintf("LayerNorm: failed to add res_x and res1: %v", err))
		}
	}
	return res
}

// goldSchmidtIter 实现 Goldschmidt 算法，用于近似 1/sqrt(x) 的同态计算
// v, y: 输入密文
// d: 迭代次数
// ckksParams: CKKS参数
// eval: Evaluator
func goldSchmidtIter(
	v, y *rlwe.Ciphertext,
	d int,
	ckksParams ckks.Parameters,
	eval *ckks.Evaluator,
) *rlwe.Ciphertext {
	// scale := y.Scale

	// x = v * y
	x, err := eval.MulRelinNew(v, y)
	if err != nil {
		panic(fmt.Sprintf("goldSchmidtIter: failed to multiply v and y: %v", err))
	}
	if err := eval.Rescale(x, x); err != nil {
		panic(fmt.Sprintf("goldSchmidtIter: failed to rescale x: %v", err))
	}

	// h = y * 0.5
	h, err := eval.MulRelinNew(y, 0.5)
	if err != nil {
		panic(fmt.Sprintf("goldSchmidtIter: failed to multiply y and 0.5: %v", err))
	}
	if err := eval.Rescale(h, h); err != nil {
		panic(fmt.Sprintf("goldSchmidtIter: failed to rescale h: %v", err))
	}

	for i := 0; i < d; i++ {
		// r = x * h
		r, err := eval.MulRelinNew(x, h)
		if err != nil {
			panic(fmt.Sprintf("goldSchmidtIter: failed to multiply x and h: %v", err))
		}
		if err := eval.Rescale(r, r); err != nil {
			panic(fmt.Sprintf("goldSchmidtIter: failed to rescale r: %v", err))
		}
		// r.Scale = scale

		// r = 0.5 - r
		err = eval.Mul(r, -1, r)
		if err != nil {
			panic(fmt.Sprintf("goldSchmidtIter: failed to negate r: %v", err))
		}
		err = eval.Add(r, 0.5, r)
		if err != nil {
			panic(fmt.Sprintf("goldSchmidtIter: failed to add 0.5 to r: %v", err))
		}

		// x = x + x*r
		tmp, err := eval.MulRelinNew(x, r)
		if err != nil {
			panic(fmt.Sprintf("goldSchmidtIter: failed to multiply x and r: %v", err))
		}
		if err := eval.Rescale(tmp, tmp); err != nil {
			panic(fmt.Sprintf("goldSchmidtIter: failed to rescale tmp: %v", err))
		}
		// x.Scale = scale
		// tmp.Scale = scale
		err = eval.Add(x, tmp, x)
		if err != nil {
			panic(fmt.Sprintf("goldSchmidtIter: failed to add tmp to x: %v", err))
		}

		// h = h + h*r
		tmp2, err := eval.MulRelinNew(h, r)
		if err != nil {
			panic(fmt.Sprintf("goldSchmidtIter: failed to multiply h and r: %v", err))
		}
		if err := eval.Rescale(tmp2, tmp2); err != nil {
			panic(fmt.Sprintf("goldSchmidtIter: failed to rescale tmp2: %v", err))
		}
		// h.Scale = scale
		// tmp2.Scale = scale
		err = eval.Add(h, tmp2, h)
		if err != nil {
			panic(fmt.Sprintf("goldSchmidtIter: failed to add tmp2 to h: %v", err))
		}
	}

	// h = h * 2.0
	err = eval.Mul(h, 2, h)
	if err != nil {
		panic(fmt.Sprintf("goldSchmidtIter: failed to multiply h and 2.0: %v", err))
	}

	return h
}
