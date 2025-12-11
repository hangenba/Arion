package he

import (
	"Arion/configs"
	"sync"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func DecryptCiphertextMatricesMT(
	ctMats *CiphertextMatrices,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	dec *rlwe.Decryptor,
	ecd *ckks.Encoder,
	numThreads int,
) ([][]float64, error) {

	num := len(ctMats.Ciphertexts)
	result := make([][]float64, num)

	var wg sync.WaitGroup
	chunkSize := (num + numThreads - 1) / numThreads

	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := (t + 1) * chunkSize
		if end > num {
			end = num
		}
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			localDec := dec.ShallowCopy() // 每个线程一个 Decryptor
			localEcd := ecd.ShallowCopy() // 每个线程一个 Encoder
			for i := start; i < end; i++ {
				value := make([]float64, modelParams.NumBatch*modelParams.NumRow)
				pt := localDec.DecryptNew(ctMats.Ciphertexts[i])
				if err := localEcd.Decode(pt, value); err != nil {
					panic(err)
				}
				result[i] = value
			}
		}(start, end)
	}

	wg.Wait()
	return result, nil
}
