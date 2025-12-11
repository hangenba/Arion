package layernorm

import (
	"Arion/configs"
	"Arion/pkg/he"
	"fmt"
	"math"
	"sync"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"gonum.org/v1/gonum/mat"
)

func LayerNormSelfAttentionOutputMT(
	ctX *he.CiphertextMatrices,
	gammaMat, betaMat mat.Matrix,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	ecd *ckks.Encoder,
	eval *ckks.Evaluator,
	numThreads int,
) (*he.CiphertextMatrices, error) {
	numCt := len(ctX.Ciphertexts)
	ctBatch := modelParams.NumBatch
	ctRealRow := modelParams.NumRealRow

	// 1. 并行计算ctSum = sum(x_i), 并放缩x_i
	ctSum := ckks.NewCiphertext(ckksParams, ctX.Ciphertexts[0].Degree(), ctX.Ciphertexts[0].Level())
	ctSumMu := &sync.Mutex{}
	var wg sync.WaitGroup
	chunkSize := (numCt + numThreads - 1) / numThreads

	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := min((t+1)*chunkSize, numCt)
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			localEval := eval.ShallowCopy()
			localSum := ckks.NewCiphertext(ckksParams, ctX.Ciphertexts[0].Degree(), ctX.Ciphertexts[0].Level())
			for i := start; i < end; i++ {
				localEval.Add(localSum, ctX.Ciphertexts[i], localSum)
				_ = localEval.Mul(ctX.Ciphertexts[i], numCt, ctX.Ciphertexts[i])
			}
			ctSumMu.Lock()
			eval.Add(ctSum, localSum, ctSum)
			ctSumMu.Unlock()
		}(start, end)
	}
	wg.Wait()

	// 2. 并行计算ctXSubSumSquare[i] = d*x_i - sum(x), 并求平方和
	ctXSubSumSquare := make([]*rlwe.Ciphertext, numCt)
	ctSumSquare := ckks.NewCiphertext(ckksParams, ctX.Ciphertexts[0].Degree(), ctX.Ciphertexts[0].Level())
	ctSumSquareMu := &sync.Mutex{}

	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := min((t+1)*chunkSize, numCt)
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			localEval := eval.ShallowCopy()
			localSumSquare := ckks.NewCiphertext(ckksParams, ctX.Ciphertexts[0].Degree(), ctX.Ciphertexts[0].Level())
			for i := start; i < end; i++ {
				ctTmp, _ := localEval.SubNew(ctX.Ciphertexts[i], ctSum)
				_ = localEval.MulRelinThenAdd(ctTmp, ctTmp, localSumSquare)
				ctXSubSumSquare[i] = ctTmp
			}
			ctSumSquareMu.Lock()
			eval.Add(ctSumSquare, localSumSquare, ctSumSquare)
			ctSumSquareMu.Unlock()
		}(start, end)
	}
	wg.Wait()

	// _ = eval.Rescale(ctSumSquare, ctSumSquare)
	// _ = eval.MulRelin(ctSumSquare, 1/float64(numCt*numCt), ctSumSquare)
	// _ = eval.Rescale(ctSumSquare, ctSumSquare)

	// === invSqrtVar 单独线程计算 ===
	var invSqrtVar *rlwe.Ciphertext
	var invDone sync.WaitGroup
	invDone.Add(1)
	go func() {
		defer invDone.Done()
		localEval := eval.ShallowCopy()
		_ = localEval.Rescale(ctSumSquare, ctSumSquare)
		_ = localEval.MulRelin(ctSumSquare, 1/float64(numCt*numCt), ctSumSquare)
		_ = localEval.Rescale(ctSumSquare, ctSumSquare)
		invSqrtVar = InvertSqrtByChebyshevAndNewtonIter(ctSumSquare, &ckksParams, localEval, modelParams.InvSqrtMinValue1*float64(numCt), modelParams.InvSqrtMaxValue1*float64(numCt), modelParams.InvSqrtDegree1, modelParams.InvSqrtIter1)
	}()

	// === 并行计算乘以 gamma/sqrt(d) (用numThreads-1个线程) ===
	preNormResults := make([]*rlwe.Ciphertext, numCt)
	chunkSize = (numCt + (numThreads - 2)) / (numThreads - 1)
	for t := 0; t < numThreads-1; t++ {
		start := t * chunkSize
		end := min((t+1)*chunkSize, numCt)
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			localEval := eval.ShallowCopy()
			for i := start; i < end; i++ {
				baseVal := float64(gammaMat.At(i, 0)) / math.Sqrt(float64(numCt))
				plainVector := make([]float64, ctBatch*ctRealRow)
				for j := 0; j < len(plainVector); j++ {
					plainVector[j] = baseVal
				}
				ctMulGamma, _ := localEval.MulRelinNew(ctXSubSumSquare[i], plainVector)
				_ = localEval.Rescale(ctMulGamma, ctMulGamma)
				preNormResults[i] = ctMulGamma
			}
		}(start, end)
	}
	wg.Wait()

	// 等 invSqrtVar 线程完成
	invDone.Wait()

	// === 最后所有线程乘以 invSqrtVar + 加 beta ===
	newCiphertexts := make([]*rlwe.Ciphertext, numCt)
	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := min((t+1)*chunkSize, numCt)
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			localEval := eval.ShallowCopy()
			for i := start; i < end; i++ {
				ctMulInvSqrtVar, _ := localEval.MulRelinNew(preNormResults[i], invSqrtVar)
				_ = localEval.Rescale(ctMulInvSqrtVar, ctMulInvSqrtVar)

				biasVal := betaMat.At(i, 0)
				biasVec := make([]float64, ctBatch*ctRealRow)
				for j := 0; j < len(biasVec); j++ {
					biasVec[j] = biasVal
				}
				_ = localEval.Add(ctMulInvSqrtVar, biasVec, ctMulInvSqrtVar)

				newCiphertexts[i] = ctMulInvSqrtVar
			}
		}(start, end)
	}
	wg.Wait()

	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumRow:      ctX.NumRow,
		NumCol:      ctX.NumCol,
		NumBatch:    ctX.NumBatch,
	}, nil
}

func LayerNormOutputMT(
	ctX *he.CiphertextMatrices,
	gammaMat, betaMat mat.Matrix,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	ecd *ckks.Encoder,
	eval *ckks.Evaluator,
	numThreads int,
) (*he.CiphertextMatrices, error) {

	numCt := len(ctX.Ciphertexts)
	ctBatch := modelParams.NumBatch
	ctRealRow := modelParams.NumRealRow

	if numCt != modelParams.NumCol {
		return nil, fmt.Errorf("LayerNorm expects %d ciphertexts, got %d", modelParams.NumCol, numCt)
	}

	var wg sync.WaitGroup
	chunkSize := (numCt + numThreads - 1) / numThreads

	// 1. 并行计算 sum(x_i) 并放缩每个x_i
	ctSum := ckks.NewCiphertext(ckksParams, ctX.Ciphertexts[0].Degree(), ctX.Ciphertexts[0].Level())
	ctSumMu := &sync.Mutex{}

	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := min((t+1)*chunkSize, numCt)
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			localEval := eval.ShallowCopy()
			localSum := ckks.NewCiphertext(ckksParams, ctX.Ciphertexts[0].Degree(), ctX.Ciphertexts[0].Level())
			for i := start; i < end; i++ {
				localEval.Add(localSum, ctX.Ciphertexts[i], localSum)
				_ = localEval.Mul(ctX.Ciphertexts[i], numCt, ctX.Ciphertexts[i])
			}
			ctSumMu.Lock()
			eval.Add(ctSum, localSum, ctSum)
			ctSumMu.Unlock()
		}(start, end)
	}
	wg.Wait()

	// 2. 并行计算 (d*x_i - SUM(X))^2 并累计平方和
	ctXSubSumSquare := make([]*rlwe.Ciphertext, numCt)
	ctSumSquare := ckks.NewCiphertext(ckksParams, ctX.Ciphertexts[0].Degree(), ctX.Ciphertexts[0].Level())
	ctSumSquareMu := &sync.Mutex{}

	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := min((t+1)*chunkSize, numCt)
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			localEval := eval.ShallowCopy()
			localSumSquare := ckks.NewCiphertext(ckksParams, ctX.Ciphertexts[0].Degree(), ctX.Ciphertexts[0].Level())
			localSumSquare.Scale = rlwe.NewScale(ckksParams.Q()[localSumSquare.Level()])
			for i := start; i < end; i++ {
				ctTmp, _ := localEval.SubNew(ctX.Ciphertexts[i], ctSum)
				_ = localEval.MulRelinThenAdd(ctTmp, ctTmp, localSumSquare)
				ctXSubSumSquare[i] = ctTmp
			}
			ctSumSquareMu.Lock()
			eval.Add(ctSumSquare, localSumSquare, ctSumSquare)
			ctSumSquareMu.Unlock()
		}(start, end)
	}
	wg.Wait()

	// _ = eval.Rescale(ctSumSquare, ctSumSquare)
	// _ = eval.MulRelin(ctSumSquare, 1/float64(numCt*numCt*numCt), ctSumSquare)
	// _ = eval.Rescale(ctSumSquare, ctSumSquare)

	// === invSqrtVar 单独线程 ===
	var invSqrtVar *rlwe.Ciphertext
	var invDone sync.WaitGroup
	invDone.Add(1)
	go func() {
		defer invDone.Done()
		localEval := eval.ShallowCopy()
		_ = localEval.Rescale(ctSumSquare, ctSumSquare)
		_ = localEval.MulRelin(ctSumSquare, 1/float64(numCt*numCt*numCt), ctSumSquare)
		_ = localEval.Rescale(ctSumSquare, ctSumSquare)
		invSqrtVar = InvertSqrtByChebyshevAndNewtonIter(ctSumSquare, &ckksParams, localEval, modelParams.InvSqrtMinValue2, modelParams.InvSqrtMaxValue2, modelParams.InvSqrtDegree2, modelParams.InvSqrtIter2)
		invSqrtVar.Scale = rlwe.NewScale(ckksParams.Q()[invSqrtVar.Level()])
	}()

	// === 并行计算 gamma/sqrt(d) 之前部分 ===
	preNormResults := make([]*rlwe.Ciphertext, numCt)
	chunkSize = (numCt + (numThreads - 2)) / (numThreads - 1) // 预留1线程给invSqrtVar

	for t := 0; t < numThreads-1; t++ {
		start := t * chunkSize
		end := min((t+1)*chunkSize, numCt)
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			localEval := eval.ShallowCopy()
			for i := start; i < end; i++ {
				baseVal := float64(gammaMat.At(i, 0)) / float64(numCt)
				plainVector := make([]float64, ctBatch*ctRealRow)
				for j := 0; j < len(plainVector); j++ {
					plainVector[j] = baseVal
				}
				ctMulGamma, _ := localEval.MulRelinNew(ctXSubSumSquare[i], plainVector)
				localEval.Rescale(ctMulGamma, ctMulGamma)
				preNormResults[i] = ctMulGamma
			}
		}(start, end)
	}
	wg.Wait()

	// 等invSqrtVar线程完成
	invDone.Wait()

	// === 最后并行乘以 invSqrtVar 并加 beta ===
	newCiphertexts := make([]*rlwe.Ciphertext, numCt)
	chunkSize = (numCt + numThreads - 1) / numThreads
	wg = sync.WaitGroup{}

	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := min((t+1)*chunkSize, numCt)
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			localEval := eval.ShallowCopy()
			for i := start; i < end; i++ {
				ctMulInvSqrtVar, _ := localEval.MulRelinNew(preNormResults[i], invSqrtVar)
				localEval.Rescale(ctMulInvSqrtVar, ctMulInvSqrtVar)

				biasVal := betaMat.At(i, 0)
				biasVec := make([]float64, ctBatch*ctRealRow)
				for j := 0; j < len(biasVec); j++ {
					biasVec[j] = biasVal
				}
				localEval.Add(ctMulInvSqrtVar, biasVec, ctMulInvSqrtVar)

				newCiphertexts[i] = ctMulInvSqrtVar
			}
		}(start, end)
	}
	wg.Wait()

	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumRow:      ctX.NumRow,
		NumCol:      ctX.NumCol,
		NumBatch:    ctX.NumBatch,
	}, nil
}
