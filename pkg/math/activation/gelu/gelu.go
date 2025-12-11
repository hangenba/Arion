package gelu

import (
	"Arion/pkg/he"
	"Arion/pkg/math/activation"
	"math"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

// CiphertextExpChebyshev 对单个密文做Chebyshev多项式Gelu近似，返回新的密文
func CiphertextGeluChebyshev(
	ct *rlwe.Ciphertext,
	ckksParams *ckks.Parameters,
	eval *ckks.Evaluator,
) *rlwe.Ciphertext {
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
		Nodes: 255,
		A:     *bignum.NewFloat(-61, prec),
		B:     *bignum.NewFloat(136, prec),
	}
	poly := bignum.ChebyshevApproximation(Gelu, interval)

	// 这里 ApproximatePolynomialChebyshev 支持单个密文输入
	return activation.ApproximatePolynomialChebyshevSingle(ct, poly, ckksParams, eval)
}

// CiphertextMatricesGeluChebyshev 对所有密文做Chebyshev多项式Gelu近似，返回新的密文
func CiphertextMatricesGeluChebyshev(
	ctMats *he.CiphertextMatrices,
	ckksParams *ckks.Parameters,
	eval *ckks.Evaluator,
	minValue, maxValue float64,
	degree int,
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

	newCtMats := activation.ApproximatePolynomialChebyshev(ctMats, poly, ckksParams, eval)

	return newCtMats
}
