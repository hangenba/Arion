package softmax

import (
	"Arion/pkg/math/activation"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

func CiphertextInverse(
	ctX *rlwe.Ciphertext,
	ckksParams *ckks.Parameters,
	eval *ckks.Evaluator,
	minValue float64,
	maxValue float64,
	degree int,
	iter int,
) *rlwe.Ciphertext {
	// 用Chebyshev计算初值
	ctXInvInit := CiphertextInverseChebyshev(ctX, ckksParams, eval, minValue, maxValue, degree)

	// 用牛顿迭代计算1/x
	ctXInv := CiphertextInverseNewtonIteration(ctX, ctXInvInit, iter, eval)

	return ctXInv
}

func CiphertextInverseNewtonIteration(
	ctX *rlwe.Ciphertext,
	ctXInvInit *rlwe.Ciphertext,
	iter int,
	eval *ckks.Evaluator,
) *rlwe.Ciphertext {
	ctRes := ctXInvInit
	for i := 0; i < iter; i++ {
		ctTmp, err := eval.MulRelinNew(ctX, ctRes)
		if err != nil {
			panic(err)
		}
		if err := eval.Rescale(ctTmp, ctTmp); err != nil {
			panic(err)
		}

		err = eval.Sub(ctTmp, 2, ctTmp)
		if err != nil {
			panic(err)
		}
		eval.Mul(ctTmp, -1, ctTmp)

		if err := eval.MulRelin(ctRes, ctTmp, ctRes); err != nil {
			panic(err)
		}
		if err := eval.Rescale(ctRes, ctRes); err != nil {
			panic(err)
		}
	}
	return ctRes
}

// func CiphertextInverseGoldschmidt(
// 	ctX *rlwe.Ciphertext,
// 	iter int,
// 	numeratorA float64,
// 	approximationY0 float64,
// 	ckksParams *ckks.Parameters,
// 	ecd *ckks.Encoder,
// 	enc *rlwe.Encryptor,
// 	eval *ckks.Evaluator,
// ) *rlwe.Ciphertext {
// 	// 创建向量
// 	valueN := make([]float64, ckksParams.MaxSlots())
// 	for i := range valueN {
// 		valueN[i] = numeratorA
// 	}
// 	valueF := make([]float64, ckksParams.MaxSlots())
// 	for i := range valueN {
// 		valueF[i] = approximationY0
// 	}

// 	//加密成密文
// 	var err error
// 	ptN := ckks.NewPlaintext(*ckksParams, ctX.Level())
// 	if err = ecd.Encode(valueN, ptN); err != nil {
// 		panic(err)
// 	}
// 	ctN, err := enc.EncryptNew(ptN)
// 	if err != nil {
// 		panic(err)
// 	}

// 	ptF := ckks.NewPlaintext(*ckksParams, ctX.Level())
// 	if err = ecd.Encode(valueF, ptF); err != nil {
// 		panic(err)
// 	}
// 	ctF, err := enc.EncryptNew(ptF)
// 	if err != nil {
// 		panic(err)
// 	}

// 	ctD := ctX

// 	for i := 0; i < iter; i++ {
// 		//step.1 compute N_i
// 		err = eval.MulRelin(ctN, ctF, ctN)
// 		if err != nil {
// 			panic(err)
// 		}
// 		err = eval.Rescale(ctN, ctN)
// 		if err != nil {
// 			panic(err)
// 		}

// 		//step.2 compute D_i
// 		err = eval.MulRelin(ctD, ctF, ctD)
// 		if err != nil {
// 			panic(err)
// 		}
// 		err = eval.Rescale(ctD, ctD)
// 		if err != nil {
// 			panic(err)
// 		}

// 		//step.3 compute F_i=2-D_i
// 		tmpD, err := eval.MulNew(ctD, -1)
// 		if err != nil {
// 			panic(err)
// 		}
// 		err = eval.Add(tmpD, 2, ctF)
// 		if err != nil {
// 			panic(err)
// 		}
// 	}
// 	return ctN
// }

// CiphertextExpChebyshev 对单个密文做Chebyshev多项式exp近似，返回新的密文
func CiphertextInverseChebyshev(
	ct *rlwe.Ciphertext,
	ckksParams *ckks.Parameters,
	eval *ckks.Evaluator,
	minValue float64,
	maxValue float64,
	degree int,
) *rlwe.Ciphertext {
	InvX := func(x complex128) (y complex128) {
		return 1 / x
	}

	prec := ckksParams.EncodingPrecision()
	interval := bignum.Interval{
		Nodes: degree,
		A:     *bignum.NewFloat(minValue, prec),
		B:     *bignum.NewFloat(maxValue, prec),
	}
	// fmt.Println("Inverse interval:", interval.A.String(), interval.B.String())
	poly := bignum.ChebyshevApproximation(InvX, interval)
	// fmt.Println("Inverse poly coeffs:")
	// 这里 ApproximatePolynomialChebyshev 支持单个密文输入
	return activation.ApproximatePolynomialChebyshevSingle(ct, poly, ckksParams, eval)
}
