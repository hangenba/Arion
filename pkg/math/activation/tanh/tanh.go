package tanh

import (
	"Arion/pkg/he"
	"Arion/pkg/math/activation"
	"math/cmplx"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

// CiphertextTanhChebyshev 对单个密文做Chebyshev多项式Tanh近似，返回新的密文
func CiphertextTanhChebyshev(
	ct *rlwe.Ciphertext,
	ckksParams *ckks.Parameters,
	eval *ckks.Evaluator,
) *rlwe.Ciphertext {
	Tanh := func(x complex128) (y complex128) {
		return cmplx.Tanh(x)
	}

	prec := ckksParams.EncodingPrecision()
	interval := bignum.Interval{
		Nodes: 63,
		A:     *bignum.NewFloat(-5, prec),
		B:     *bignum.NewFloat(5, prec),
	}
	poly := bignum.ChebyshevApproximation(Tanh, interval)

	// 这里 ApproximatePolynomialChebyshev 支持单个密文输入
	return activation.ApproximatePolynomialChebyshevSingle(ct, poly, ckksParams, eval)
}

// CiphertextTanhChebyshev 对单个密文做Chebyshev多项式Tanh近似，返回新的密文
func CiphertextMatricesTanhChebyshev(
	ctMats *he.CiphertextMatrices,
	ckksParams *ckks.Parameters,
	eval *ckks.Evaluator,
) *he.CiphertextMatrices {
	Tanh := func(x complex128) (y complex128) {
		return cmplx.Tanh(x)
	}

	prec := ckksParams.EncodingPrecision()
	interval := bignum.Interval{
		Nodes: 63,
		A:     *bignum.NewFloat(-10, prec),
		B:     *bignum.NewFloat(10, prec),
	}
	poly := bignum.ChebyshevApproximation(Tanh, interval)

	newCtMats := activation.ApproximatePolynomialChebyshev(ctMats, poly, ckksParams, eval)

	return newCtMats
}
