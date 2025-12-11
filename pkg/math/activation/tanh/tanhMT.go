package tanh

import (
	"Arion/pkg/he"
	"Arion/pkg/math/activation"
	"math/cmplx"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

// CiphertextTanhChebyshev 对单个密文做Chebyshev多项式Tanh近似，返回新的密文
func CiphertextMatricesTanhChebyshevMT(
	ctMats *he.CiphertextMatrices,
	ckksParams *ckks.Parameters,
	eval *ckks.Evaluator,
	numThreads int,
) *he.CiphertextMatrices {
	Tanh := func(x complex128) (y complex128) {
		return cmplx.Tanh(x)
	}

	prec := ckksParams.EncodingPrecision()
	interval := bignum.Interval{
		Nodes: 127,
		A:     *bignum.NewFloat(-20, prec),
		B:     *bignum.NewFloat(20, prec),
	}
	poly := bignum.ChebyshevApproximation(Tanh, interval)

	newCtMats := activation.ApproximatePolynomialChebyshevMT(ctMats, poly, ckksParams, eval, numThreads)

	return newCtMats
}
