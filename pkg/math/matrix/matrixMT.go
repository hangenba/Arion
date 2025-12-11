package matrix

import (
	"Arion/configs"
	"Arion/pkg/he"
	"fmt"
	"sync"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"gonum.org/v1/gonum/mat"
)

func CiphertextMatricesMultiplyWeightAndAddBiasMT(
	ctMatrices *he.CiphertextMatrices,
	ptWeight mat.Matrix,
	ptBias mat.Matrix,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	eval *ckks.Evaluator,
	numThreads int,
) (*he.CiphertextMatrices, error) {

	// ctRows := ctMatrices.NumRow
	ctCols := ctMatrices.NumCol
	ctBatch := ctMatrices.NumBatch
	ctRealRows := modelParams.NumRealRow

	ptRows, ptCols := ptWeight.Dims()
	biasRows, biasCols := ptBias.Dims()

	if ctCols != ptRows {
		return nil, fmt.Errorf("ciphertext Matrices columns %d do not match plaintext weight rows %d", ctCols, ptRows)
	}
	if biasCols != 1 {
		return nil, fmt.Errorf("ptBias must be shape (d,1), got (d,%d)", biasCols)
	}
	if ptCols != biasRows {
		return nil, fmt.Errorf("output depth %d does not match bias length %d", ptCols, biasRows)
	}

	newCiphertexts := make([]*rlwe.Ciphertext, ptCols)
	// for i := 0; i < len(ctMatrices.Ciphertexts); i++ {
	// 	ctMatrices.Ciphertexts[i].Scale = ckksParams.DefaultScale()
	// }

	// runtime.GOMAXPROCS(numThreads)
	chunkSize := (ptCols + numThreads - 1) / numThreads

	var wg sync.WaitGroup
	errChan := make(chan error, numThreads)

	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := (t + 1) * chunkSize
		if end > ptCols {
			end = ptCols
		}
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			localEval := eval.ShallowCopy()
			// localctMatrices := ctMatrices.CopyNew()

			for i := start; i < end; i++ {
				ct, _ := localEval.MulRelinNew(ctMatrices.Ciphertexts[0], ptWeight.At(0, i))
				// === MulThenAdd 累加部分 ===
				for j := 1; j < ptRows; j++ {
					// ptMulNumSlice := make([]float64, ctBatch*ctRealRows)
					// for k := range ptMulNumSlice {
					// 	ptMulNumSlice[k] = ptWeight.At(j, i)
					// }

					// ctMatrices.Ciphertexts[j].Scale = ckksParams.DefaultScale()
					err := localEval.MulRelinThenAdd(ctMatrices.Ciphertexts[j], ptWeight.At(j, i), ct)
					if err != nil {
						errChan <- fmt.Errorf("MulThenAdd error at output %d, input %d: %w", i, j, err)
						return
					}
				}

				// === Rescale ===
				if err := localEval.Rescale(ct, ct); err != nil {
					errChan <- fmt.Errorf("rescale error at output %d: %w", i, err)
					return
				}

				// === Add Bias ===
				biasVal := ptBias.At(i, 0)
				biasVec := make([]float64, ctBatch*ctRealRows)
				for k := range biasVec {
					biasVec[k] = biasVal
				}

				// ct.Scale = ckksParams.DefaultScale()
				ctWithBias, err := localEval.AddNew(ct, biasVec)
				if err != nil {
					errChan <- fmt.Errorf("AddBias error at output %d: %w", i, err)
					return
				}

				newCiphertexts[i] = ctWithBias
			}
		}(start, end)
	}

	wg.Wait()
	close(errChan)

	if len(errChan) > 0 {
		return nil, <-errChan
	}

	// ctScale := &newCiphertexts[0].Scale.Value // We need to access the pointer in order for it to display correctly in the command line.
	// fmt.Printf("CiphertextMatricesMultiplyWeightAndAddBiasMT Scale rescaling: %f\n", ctScale)

	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ctMatrices.NumBatch,
		NumRow:      ctMatrices.NumRow,
		NumCol:      ptCols,
	}, nil
}

func RotateCiphertextMatricesHoistingMT(
	ctMatrices *he.CiphertextMatrices,
	stepsSlice []int,
	eval *ckks.Evaluator,
	numThreads int,
) []*he.CiphertextMatrices {
	numCols := len(ctMatrices.Ciphertexts)
	numSteps := len(stepsSlice)

	// 初始化结果，每个步长一个 CiphertextMatrices
	result := make([]*he.CiphertextMatrices, numSteps)
	for i := 0; i < numSteps; i++ {
		result[i] = &he.CiphertextMatrices{
			Ciphertexts: make([]*rlwe.Ciphertext, numCols),
			NumBatch:    ctMatrices.NumBatch,
			NumRow:      ctMatrices.NumRow,
			NumCol:      ctMatrices.NumCol,
		}
	}

	chunkSize := (numCols + numThreads - 1) / numThreads

	var wg sync.WaitGroup
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
			localEval := eval.ShallowCopy() // 每个线程使用自己的 evaluator
			for colIdx := start; colIdx < end; colIdx++ {
				ct := ctMatrices.Ciphertexts[colIdx]
				rotMap, err := localEval.RotateHoistedNew(ct, stepsSlice)
				if err != nil {
					panic(err)
				}
				for stepIdx, step := range stepsSlice {
					result[stepIdx].Ciphertexts[colIdx] = rotMap[step]
				}
			}
		}(start, end)
	}
	wg.Wait()

	return result
}

func RotateCiphertextMatricesMT(
	ctMatrices *he.CiphertextMatrices,
	stepsSlice []int,
	eval *ckks.Evaluator,
	numThreads int,
) []*he.CiphertextMatrices {
	numCols := len(ctMatrices.Ciphertexts)
	numSteps := len(stepsSlice)

	// 初始化结果，每个步长一个 CiphertextMatrices
	result := make([]*he.CiphertextMatrices, numSteps)
	for i := 0; i < numSteps; i++ {
		result[i] = &he.CiphertextMatrices{
			Ciphertexts: make([]*rlwe.Ciphertext, numCols),
			NumBatch:    ctMatrices.NumBatch,
			NumRow:      ctMatrices.NumRow,
			NumCol:      ctMatrices.NumCol,
		}
	}

	chunkSize := (numCols + numThreads - 1) / numThreads

	var wg sync.WaitGroup
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
			localEval := eval.ShallowCopy() // 每个线程使用自己的 evaluator
			for colIdx := start; colIdx < end; colIdx++ {
				ct := ctMatrices.Ciphertexts[colIdx]
				for stepIdx, step := range stepsSlice {
					rotTmp, err := localEval.RotateNew(ct, step)
					if err != nil {
						panic(err)
					}
					result[stepIdx].Ciphertexts[colIdx] = rotTmp
				}
			}
		}(start, end)
	}
	wg.Wait()

	return result
}

func CiphertextMatricesAddCiphertextMatricesMT(
	ctMats1 *he.CiphertextMatrices,
	ctMats2 *he.CiphertextMatrices,
	eval *ckks.Evaluator,
	numThreads int,
) (*he.CiphertextMatrices, error) {

	if ctMats1.NumBatch != ctMats2.NumBatch ||
		ctMats1.NumRow != ctMats2.NumRow ||
		ctMats1.NumCol != ctMats2.NumCol ||
		len(ctMats1.Ciphertexts) != len(ctMats2.Ciphertexts) {
		return nil, fmt.Errorf("CiphertextMatricesAddCiphertextMatricesMT: 维度不匹配")
	}

	numCts := len(ctMats1.Ciphertexts)
	newCiphertexts := make([]*rlwe.Ciphertext, numCts)

	var wg sync.WaitGroup
	chunkSize := (numCts + numThreads - 1) / numThreads // 每线程处理 chunkSize 个密文

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
			localEval := eval.ShallowCopy() // 每个线程一个 evaluator 副本
			for i := start; i < end; i++ {
				ct, err := localEval.AddNew(ctMats1.Ciphertexts[i], ctMats2.Ciphertexts[i])
				if err != nil {
					panic(fmt.Errorf("CiphertextMatricesAddCiphertextMatricesMT: 第%d个密文加法失败: %v", i, err))
				}
				newCiphertexts[i] = ct
			}
		}(start, end)
	}

	wg.Wait()

	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ctMats1.NumBatch,
		NumRow:      ctMats1.NumRow,
		NumCol:      ctMats1.NumCol,
	}, nil
}
