package silu

import (
	"Arion/pkg/he"
	"Arion/pkg/math/activation"
	"math"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

// CiphertextExpChebyshev 对单个密文做Chebyshev多项式Gelu近似，返回新的密文
func CiphertextSiLUChebyshev(
	ct *rlwe.Ciphertext,
	ckksParams *ckks.Parameters,
	eval *ckks.Evaluator,
) *rlwe.Ciphertext {
	SiLU := func(x complex128) complex128 {
		realX := real(x)
		sig := 1.0 / (1.0 + math.Exp(-realX))
		yReal := realX * sig
		return complex(yReal, 0)
	}

	prec := ckksParams.EncodingPrecision()
	interval := bignum.Interval{
		Nodes: 255,
		A:     *bignum.NewFloat(-61, prec),
		B:     *bignum.NewFloat(136, prec),
	}
	poly := bignum.ChebyshevApproximation(SiLU, interval)

	// 这里 ApproximatePolynomialChebyshev 支持单个密文输入
	return activation.ApproximatePolynomialChebyshevSingle(ct, poly, ckksParams, eval)
}

// CiphertextMatricesGeluChebyshev 对所有密文做Chebyshev多项式Gelu近似，返回新的密文
func CiphertextMatricesSiLUChebyshev(
	ctMats *he.CiphertextMatrices,
	ckksParams *ckks.Parameters,
	eval *ckks.Evaluator,
	minValue, maxValue float64,
	degree int,
) *he.CiphertextMatrices {
	SiLU := func(x complex128) complex128 {
		realX := real(x)
		sig := 1.0 / (1.0 + math.Exp(-realX))
		yReal := realX * sig
		return complex(yReal, 0)
	}

	prec := ckksParams.EncodingPrecision()
	interval := bignum.Interval{
		Nodes: degree,
		A:     *bignum.NewFloat(minValue, prec),
		B:     *bignum.NewFloat(maxValue, prec),
	}
	poly := bignum.ChebyshevApproximation(SiLU, interval)

	newCtMats := activation.ApproximatePolynomialChebyshev(ctMats, poly, ckksParams, eval)

	return newCtMats
}
