package activation

import (
	"Arion/pkg/he"
	"sync"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/polynomial"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

func ApproximatePolynomialChebyshevMT(
	ctMats *he.CiphertextMatrices,
	poly bignum.Polynomial,
	ckksParams *ckks.Parameters,
	eval *ckks.Evaluator,
	numThreads int,
) *he.CiphertextMatrices {

	numCts := len(ctMats.Ciphertexts)
	newCiphertexts := make([]*rlwe.Ciphertext, numCts)

	scalarmul, scalaradd := poly.ChangeOfBasis()

	var wg sync.WaitGroup
	chunkSize := (numCts + numThreads - 1) / numThreads

	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := (t + 1) * chunkSize
		if end > numCts {
			end = numCts
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			localEval := eval.ShallowCopy()
			polyEval := polynomial.NewEvaluator(*ckksParams, localEval)

			for i := start; i < end; i++ {
				res, err := localEval.MulNew(ctMats.Ciphertexts[i], scalarmul)
				if err != nil {
					panic(err)
				}
				if err = localEval.Add(res, scalaradd, res); err != nil {
					panic(err)
				}
				if err = localEval.Rescale(res, res); err != nil {
					panic(err)
				}

				res, err = polyEval.Evaluate(res, poly, ckksParams.DefaultScale())
				if err != nil {
					panic(err)
				}
				newCiphertexts[i] = res
			}
		}(start, end)
	}
	wg.Wait()

	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ctMats.NumBatch,
		NumRow:      ctMats.NumRow,
		NumCol:      ctMats.NumCol,
	}
}
