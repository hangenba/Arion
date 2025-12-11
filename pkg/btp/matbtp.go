package btp

import (
	"Arion/pkg/he"
	"fmt"
	"sync"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

func CiphertextMatricesBootstrapping(
	ctX *he.CiphertextMatrices,
	btpEval *bootstrapping.Evaluator,
) (*he.CiphertextMatrices, error) {
	// This function is a placeholder for the bootstrapping logic.
	newCiphertext := make([]*rlwe.Ciphertext, len(ctX.Ciphertexts))
	for i := 0; i < ctX.NumCol; i++ {
		ct, err := btpEval.Bootstrap(ctX.Ciphertexts[i])
		if err != nil {
			panic(err)
		}
		newCiphertext[i] = ct
	}
	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertext,
		NumRow:      ctX.NumRow,
		NumCol:      ctX.NumCol,
		NumBatch:    ctX.NumBatch,
	}, nil
}

func CiphertextMatricesBootstrappingMT(
	ctX *he.CiphertextMatrices,
	btpEval *bootstrapping.Evaluator,
	btpParams bootstrapping.Parameters,
	numThreads int,
) (*he.CiphertextMatrices, error) {
	numCols := ctX.NumCol
	newCiphertext := make([]*rlwe.Ciphertext, numCols)

	var wg sync.WaitGroup
	chunkSize := (numCols + numThreads - 1) / numThreads // 动态分配chunk

	var mu sync.Mutex

	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := (t + 1) * chunkSize
		if end > numCols {
			end = numCols
		}
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			localBtpEval := DeepCopyBootstrapEvaluator(btpEval, btpParams) // 每个线程使用自己的 evaluator
			// localBtpEval := btpEval.ShallowCopy()
			for i := start; i < end; i++ {
				// fmt.Println(i)
				ct, _ := localBtpEval.Bootstrap(ctX.Ciphertexts[i])

				mu.Lock()
				newCiphertext[i] = ct
				mu.Unlock()
			}
		}(start, end)
	}

	wg.Wait()

	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertext,
		NumRow:      ctX.NumRow,
		NumCol:      ctX.NumCol,
		NumBatch:    ctX.NumBatch,
	}, nil
}

func DeepCopyBootstrapEvaluator(
	origEval *bootstrapping.Evaluator,
	btpParams bootstrapping.Parameters,
) *bootstrapping.Evaluator {

	// 生成新的 Evaluator 实例（底层 buffers 独立）
	newEval, err := bootstrapping.NewEvaluator(btpParams, origEval.EvaluationKeys)
	if err != nil {
		panic(fmt.Sprintf("Failed to DeepCopy Evaluator: %v", err))
	}
	return newEval
}
