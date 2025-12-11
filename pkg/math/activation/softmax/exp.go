package softmax

import (
	"Arion/pkg/he"
	"Arion/pkg/math/activation"
	"math"
	"math/cmplx"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

// CiphertextMatricesExp 对 CiphertextMatrices 中所有密文做指数近似（多项式近似exp），返回新的 CiphertextMatrices
// 这里只做简单的多项式近似：exp(x) ≈ (x*0.0078125+1)^128
func CiphertextMatricesExp(
	ctMats *he.CiphertextMatrices,
	eval *ckks.Evaluator,
) *he.CiphertextMatrices {
	newCiphertexts := make([]*rlwe.Ciphertext, len(ctMats.Ciphertexts))
	for i, ct := range ctMats.Ciphertexts {
		// exp(x) ≈ (x*0.0078125+1)^128

		//step1.compute x*0.0078125
		ct2, err := eval.MulRelinNew(ct, 0.0078125)
		if err != nil {
			panic(err)
		}
		err = eval.Rescale(ct2, ct2)
		if err != nil {
			panic(err)
		}

		//step2. compute x*0.0078125+1
		err = eval.Add(ct2, 1, ct2)
		if err != nil {
			panic(err)
		}
		for j := 0; j < int(math.Log2(128)); j++ {
			err = eval.MulRelin(ct2, ct2, ct2)
			if err != nil {
				panic(err)
			}
			err = eval.Rescale(ct2, ct2)
			if err != nil {
				panic(err)
			}
		}
		newCiphertexts[i] = ct2
	}
	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ctMats.NumBatch,
		NumRow:      ctMats.NumRow,
		NumCol:      ctMats.NumCol,
	}
}

// CiphertextExp 对单个密文做指数近似（多项式近似exp），返回新的密文
// 这里只做简单的多项式近似：exp(x) ≈ (x*0.0078125+1)^128
func CiphertextExp(
	ct *rlwe.Ciphertext,
	eval *ckks.Evaluator,
) *rlwe.Ciphertext {
	// step1. compute x*0.0078125
	ct2, err := eval.MulRelinNew(ct, 0.0078125)
	if err != nil {
		panic(err)
	}
	err = eval.Rescale(ct2, ct2)
	if err != nil {
		panic(err)
	}

	// step2. compute x*0.0078125+1
	err = eval.Add(ct2, 1, ct2)
	if err != nil {
		panic(err)
	}
	for j := 0; j < int(math.Log2(128)); j++ {
		err = eval.MulRelin(ct2, ct2, ct2)
		if err != nil {
			panic(err)
		}
		err = eval.Rescale(ct2, ct2)
		if err != nil {
			panic(err)
		}
	}
	return ct2
}

func CiphertextMatricesExpChebyshev(
	ctMats *he.CiphertextMatrices,
	ckksParams *ckks.Parameters,
	eval *ckks.Evaluator,
) *he.CiphertextMatrices {
	Exp := func(x complex128) (y complex128) {
		return cmplx.Exp(x)
	}

	prec := ckksParams.EncodingPrecision()
	interval := bignum.Interval{
		Nodes: 63,
		A:     *bignum.NewFloat(-60, prec),
		B:     *bignum.NewFloat(15, prec),
	}
	poly := bignum.ChebyshevApproximation(Exp, interval)

	newCtMats := activation.ApproximatePolynomialChebyshev(ctMats, poly, ckksParams, eval)

	return newCtMats
}

// CiphertextExpChebyshev 对单个密文做Chebyshev多项式exp近似，返回新的密文
func CiphertextExpChebyshev(
	ct *rlwe.Ciphertext,
	ckksParams *ckks.Parameters,
	eval *ckks.Evaluator,
	minValue, maxValue float64,
	degree int,
) *rlwe.Ciphertext {
	Exp := func(x complex128) (y complex128) {
		return cmplx.Exp(x)
	}

	prec := ckksParams.EncodingPrecision()
	interval := bignum.Interval{
		Nodes: degree,
		A:     *bignum.NewFloat(minValue, prec),
		B:     *bignum.NewFloat(maxValue, prec),
	}
	poly := bignum.ChebyshevApproximation(Exp, interval)

	// 这里 ApproximatePolynomialChebyshev 支持单个密文输入
	return activation.ApproximatePolynomialChebyshevSingle(ct, poly, ckksParams, eval)
}
