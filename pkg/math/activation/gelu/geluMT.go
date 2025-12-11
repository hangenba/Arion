package gelu

import (
	"Arion/pkg/he"
	"Arion/pkg/math/activation"
	"math"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

// CiphertextMatricesGeluChebyshev 对所有密文做Chebyshev多项式Gelu近似，返回新的密文
func CiphertextMatricesGeluChebyshevMT(
	ctMats *he.CiphertextMatrices,
	ckksParams *ckks.Parameters,
	eval *ckks.Evaluator,
	minValue, maxValue float64,
	degree int,
	numThreads int,
) *he.CiphertextMatrices {
	Gelu := func(x complex128) (y complex128) {
		// Use only the real part for GELU approximation
		realX := real(x)
		inner := realX + 0.044715*math.Pow(realX, 3)
		tanhArg := math.Sqrt(2/math.Pi) * inner
		yReal := 0.5 * realX * (1 + math.Tanh(tanhArg))
		return complex(yReal, 0)
	}

	prec := ckksParams.EncodingPrecision()
	interval := bignum.Interval{
		Nodes: degree,
		A:     *bignum.NewFloat(minValue, prec),
		B:     *bignum.NewFloat(maxValue, prec),
	}
	poly := bignum.ChebyshevApproximation(Gelu, interval)

	newCtMats := activation.ApproximatePolynomialChebyshevMT(ctMats, poly, ckksParams, eval, numThreads)

	return newCtMats
}
