package he

import (
	"Arion/configs"
	"fmt"
	"runtime"
	"sync"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func EncryptInputMatricesMT(inputMatrices [][]float64, modelParams *configs.ModelParams, ckksParams ckks.Parameters, enc *rlwe.Encryptor, ecd *ckks.Encoder, numThreads int) (*CiphertextMatrices, error) {
	numMatrices := len(inputMatrices)
	ciphertexts := make([]*rlwe.Ciphertext, numMatrices)

	runtime.GOMAXPROCS(numThreads)

	chunkSize := (numMatrices + numThreads - 1) / numThreads // 向上取整
	var wg sync.WaitGroup
	errChan := make(chan error, numThreads)

	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := (t + 1) * chunkSize
		if end > numMatrices {
			end = numMatrices
		}

		if start >= end {
			continue // 防止空任务
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			encoder := ecd.ShallowCopy()
			encryptor := enc.ShallowCopy()
			pt := ckks.NewPlaintext(ckksParams, ckksParams.MaxLevel())

			for i := start; i < end; i++ {
				mat := inputMatrices[i]

				if err := encoder.Encode(mat, pt); err != nil {
					errChan <- fmt.Errorf("encode error at index %d: %w", i, err)
					return
				}

				ct, err := encryptor.EncryptNew(pt)
				if err != nil {
					errChan <- fmt.Errorf("encrypt error at index %d: %w", i, err)
					return
				}

				ciphertexts[i] = ct
			}
		}(start, end)
	}

	wg.Wait()
	close(errChan)

	if len(errChan) > 0 {
		return nil, <-errChan
	}

	return &CiphertextMatrices{
		Ciphertexts: ciphertexts,
		NumBatch:    modelParams.NumBatch,
		NumRow:      modelParams.NumRow,
		NumCol:      len(inputMatrices),
	}, nil
}

func DeepCopyCiphertextMatrices(src *CiphertextMatrices) *CiphertextMatrices {
	newCts := make([]*rlwe.Ciphertext, len(src.Ciphertexts))
	for i, ct := range src.Ciphertexts {
		newCts[i] = ct.CopyNew() // 确保是深拷贝
	}
	return &CiphertextMatrices{
		Ciphertexts: newCts,
		NumBatch:    src.NumBatch,
		NumRow:      src.NumRow,
		NumCol:      src.NumCol,
	}
}
