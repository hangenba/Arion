package bert

import (
	"Arion/configs"
	"Arion/pkg/btp"
	"Arion/pkg/he"
	"Arion/pkg/math/activation/softmax"
	"Arion/pkg/math/matrix"
	"Arion/pkg/utils"
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// ComputeQKV 计算 Q、K、V 矩阵
// 输入：ctInput 为输入密文指针，layerParams 为权重和偏置参数指针
// 输出：Q、K、V 的密文表达指针
func ComputeQKVMT(
	ctInput *he.CiphertextMatrices,
	layerParams *utils.LayerParameters,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	eval *ckks.Evaluator,
	numThreads int,
) (*he.CiphertextMatrices, *he.CiphertextMatrices, *he.CiphertextMatrices, error) {
	// Q = ctInput × weightQ + biasQ
	ctQ, err := matrix.CiphertextMatricesMultiplyWeightAndAddBiasMT(ctInput, layerParams.LayerAttentionSelfQueryWeight.T(), layerParams.LayerAttentionSelfQueryBias, modelParams, ckksParams, eval, numThreads)
	if err != nil {
		return nil, nil, nil, err
	}
	// K = ctInput × weightK + biasK
	ctK, err := matrix.CiphertextMatricesMultiplyWeightAndAddBiasMT(ctInput, layerParams.LayerAttentionSelfKeyWeight.T(), layerParams.LayerAttentionSelfKeyBias, modelParams, ckksParams, eval, numThreads)
	if err != nil {
		return nil, nil, nil, err
	}
	// V = ctInput × weightV + biasV
	ctV, err := matrix.CiphertextMatricesMultiplyWeightAndAddBiasMT(ctInput, layerParams.LayerAttentionSelfValueWeight.T(), layerParams.LayerAttentionSelfValueBias, modelParams, ckksParams, eval, numThreads)
	if err != nil {
		return nil, nil, nil, err
	}
	return ctQ, ctK, ctV, nil
}

func ComputeMultiHeadAttentionMT1(
	ctQ, ctK, ctV *he.CiphertextMatrices,
	modelParams *configs.ModelParams,
	ckksParams *ckks.Parameters,
	ecd *ckks.Encoder,
	enc *rlwe.Encryptor,
	eval *ckks.Evaluator,
	btpEval *bootstrapping.Evaluator,
	numThreads int,
) (*he.CiphertextMatrices, error) {
	// 1. Split Q, K, V by heads
	ctQsplit := matrix.SplitCiphertextMatricesByHeads(ctQ, modelParams)
	ctKsplit := matrix.SplitCiphertextMatricesByHeads(ctK, modelParams)
	ctVsplit := matrix.SplitCiphertextMatricesByHeads(ctV, modelParams)

	ctAttentionHeads := make([]*he.CiphertextMatrices, modelParams.NumHeads)

	// 2. For each head, compute attention
	for headIdx := 0; headIdx < modelParams.NumHeads; headIdx++ {
		ctAttention, _, err := ComputeAttentionMT3(ctQsplit[headIdx], ctKsplit[headIdx], ctVsplit[headIdx], modelParams, ckksParams, ecd, enc, eval, btpEval, numThreads)
		if err != nil {
			return nil, fmt.Errorf("failed to compute attention for head %d: %w", headIdx, err)
		}
		ctAttentionHeads[headIdx] = ctAttention
	}

	// 3. Merge all attention heads
	ctAttention := matrix.MergeCiphertextMatricesByHeads(ctAttentionHeads)

	return ctAttention, nil
}

func ComputeAttentionMT1(
	ctQ, ctK, ctV *he.CiphertextMatrices,
	modelParams *configs.ModelParams,
	ckksParams *ckks.Parameters,
	ecd *ckks.Encoder,
	enc *rlwe.Encryptor,
	eval *ckks.Evaluator,
	btpEval *bootstrapping.Evaluator,
	numThreads int,
) (*he.CiphertextMatrices, *rlwe.Ciphertext, error) {
	realRow := modelParams.NumRealRow
	ctBatch := ctQ.NumBatch
	ctRows := ctQ.NumRow
	ctCols := ctQ.NumCol

	babyStep := modelParams.BabyStep
	giantStep := modelParams.GiantStep
	baseLen := modelParams.NumBatch

	maskMat := utils.GenerateSpecialSquareMatrix(modelParams)

	babySteps := make([]int, babyStep)
	giantSteps := make([]int, giantStep)
	for i := 0; i < babyStep; i++ {
		babySteps[i] = i * baseLen
	}
	for i := 0; i < giantStep; i++ {
		giantSteps[i] = -i * baseLen * babyStep
	}

	ctQRotated := matrix.RotateCiphertextMatricesHoistingMT(ctQ, giantSteps, eval, numThreads)
	ctKRotated := matrix.RotateCiphertextMatricesHoistingMT(ctK, babySteps, eval, numThreads)
	ctVRotated := matrix.RotateCiphertextMatricesHoistingMT(ctV, babySteps, eval, numThreads)

	ctQKTCiphertext := make([]*rlwe.Ciphertext, ctRows)
	ctQKTAdd := ckks.NewCiphertext(*ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())

	var wg sync.WaitGroup
	var mu sync.Mutex
	chunkSize := (giantStep + numThreads - 1) / numThreads

	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := (t + 1) * chunkSize
		if end > giantStep {
			end = giantStep
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			updateMaxGoroutines() // <- 外层Goroutine启动统计
			defer wg.Done()
			localEval := eval.ShallowCopy()
			for i := start; i < end; i++ {
				// Outer i-th giant step
				localCtAddGiantStep := ckks.NewCiphertext(*ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
				localCtQKTCiphertext := make([]*rlwe.Ciphertext, babyStep)

				var innerWg sync.WaitGroup
				innerChunk := (babyStep + numThreads - 1) / numThreads
				for tInner := 0; tInner < numThreads; tInner++ {
					jStart := tInner * innerChunk
					jEnd := (tInner + 1) * innerChunk
					if jEnd > babyStep {
						jEnd = babyStep
					}
					if jStart >= jEnd {
						continue
					}
					innerWg.Add(1)
					go func(jStart, jEnd int) {
						defer innerWg.Done()
						localInnerEval := localEval.ShallowCopy()
						for j := jStart; j < jEnd; j++ {
							rowIdx := i*babyStep + j
							if rowIdx >= ctRows {
								break
							}
							ct, err := matrix.CiphertextMatricesMultiplyCiphertextMatricesThenAdd(ctQRotated[i], ctKRotated[j], ckksParams, localInnerEval)
							if err != nil {
								panic(err)
							}
							localInnerEval.Add(localCtAddGiantStep, ct, localCtAddGiantStep)
							localCtQKTCiphertext[j] = ct
						}
					}(jStart, jEnd)
				}
				innerWg.Wait()

				// 合并到全局ctQKTCiphertext
				mu.Lock()
				for j, ct := range localCtQKTCiphertext {
					rowIdx := i*babyStep + j
					if rowIdx >= ctRows {
						continue
					}
					ctQKTCiphertext[rowIdx] = ct
				}
				// 合并 GiantStep Sum
				ctAddRotate, _ := localEval.RotateNew(localCtAddGiantStep, i*baseLen*babyStep)
				eval.Add(ctQKTAdd, ctAddRotate, ctQKTAdd)
				mu.Unlock()
			}
		}(start, end)
	}
	wg.Wait()
	fmt.Printf("[Max Goroutines Reached]: %d\n", maxGoroutines)

	// 串行
	ctMean, _ := eval.MulRelinNew(ctQKTAdd, 1/float64(realRow))
	eval.Rescale(ctMean, ctMean)
	ctQKTAddRot, _ := eval.RotateHoistedNew(ctQKTAdd, giantSteps)

	constantValue := 2.08
	varScale := math.Sqrt(0.1*(1.0/float64(realRow))*constantValue) * (1.0 / float64(realRow))
	constValue := 1.25 * constantValue

	ctVarScale := ckks.NewCiphertext(*ckksParams, ctMean.Degree(), ctMean.Level())

	// --- Phase 2 并行改造 ---
	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := (t + 1) * chunkSize
		if end > giantStep {
			end = giantStep
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			localEval := eval.ShallowCopy()
			for i := start; i < end; i++ {
				localCtAddGiantStep := ckks.NewCiphertext(*ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
				var innerWg sync.WaitGroup
				innerChunk := (babyStep + numThreads - 1) / numThreads
				for tInner := 0; tInner < numThreads; tInner++ {
					jStart := tInner * innerChunk
					jEnd := (tInner + 1) * innerChunk
					if jEnd > babyStep {
						jEnd = babyStep
					}
					if jStart >= jEnd {
						continue
					}
					innerWg.Add(1)
					go func(jStart, jEnd int) {
						defer innerWg.Done()
						localInnerEval := localEval.ShallowCopy()
						for j := jStart; j < jEnd; j++ {
							rowIdx := i*babyStep + j
							if rowIdx >= ctRows {
								break
							}
							ctQKTScale, _ := localInnerEval.MulNew(ctQKTCiphertext[rowIdx], realRow)
							ctSub, _ := localInnerEval.SubNew(ctQKTScale, ctQKTAddRot[-i*baseLen*babyStep])
							maskVec := utils.ExtractAndRepeatDiagonal(maskMat, rowIdx, modelParams.NumBatch)
							for idx := range maskVec {
								maskVec[idx] *= varScale
							}
							maskVecRot := utils.RotateSliceNew(maskVec, -i*baseLen*babyStep)
							localInnerEval.MulRelin(ctSub, maskVecRot, ctSub)
							localInnerEval.Rescale(ctSub, ctSub)
							localInnerEval.MulRelin(ctSub, ctSub, ctSub)
							localInnerEval.Rescale(ctSub, ctSub)
							localInnerEval.Add(localCtAddGiantStep, ctSub, localCtAddGiantStep)
						}
					}(jStart, jEnd)
				}
				innerWg.Wait()
				mu.Lock()
				ctAddRotate, _ := localEval.RotateNew(localCtAddGiantStep, i*baseLen*babyStep)
				eval.Add(ctVarScale, ctAddRotate, ctVarScale)
				mu.Unlock()
			}
		}(start, end)
	}
	wg.Wait()

	ctApproxStd, _ := eval.AddNew(ctVarScale, constValue)
	ctApproxMax, _ := eval.AddNew(ctApproxStd, ctMean)
	ctApproxMaxRot, _ := eval.RotateHoistedNew(ctApproxMax, giantSteps)

	ctExpStore := make([]*rlwe.Ciphertext, ctRows)
	ctAddAllbyRow := ckks.NewCiphertext(*ckksParams, ctApproxMax.Degree(), ctApproxMax.Level())

	// Phase 3: 并行计算 ctAddAllbyRow 并保存 ctExp
	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := (t + 1) * chunkSize
		if end > giantStep {
			end = giantStep
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			localEval := eval.ShallowCopy()
			for i := start; i < end; i++ {
				localCtAddGiantStep := ckks.NewCiphertext(*ckksParams, ctApproxMax.Degree(), ctApproxMax.Level())
				localCtExpStore := make([]*rlwe.Ciphertext, babyStep)

				var innerWg sync.WaitGroup
				innerChunk := (babyStep + numThreads - 1) / numThreads
				localCtAddGiantStepLocals := make([]*rlwe.Ciphertext, numThreads) // 每个inner线程一个累加器

				for tInner := 0; tInner < numThreads; tInner++ {
					localCtAddGiantStepLocals[tInner] = ckks.NewCiphertext(*ckksParams, ctApproxMax.Degree(), ctApproxMax.Level())
				}

				for tInner := 0; tInner < numThreads; tInner++ {
					jStart := tInner * innerChunk
					jEnd := (tInner + 1) * innerChunk
					if jEnd > babyStep {
						jEnd = babyStep
					}
					if jStart >= jEnd {
						continue
					}
					innerWg.Add(1)
					go func(tInner, jStart, jEnd int) {
						defer innerWg.Done()
						localInnerEval := localEval.ShallowCopy()
						for j := jStart; j < jEnd; j++ {
							rowIdx := i*babyStep + j
							if rowIdx >= ctRows {
								break
							}
							ctSub, _ := localInnerEval.SubNew(ctQKTCiphertext[rowIdx], ctApproxMaxRot[-i*baseLen*babyStep])
							ctExp := softmax.CiphertextExpChebyshev(ctSub, ckksParams, localInnerEval, modelParams.ExpMinValue, modelParams.ExpMaxValue, modelParams.ExpDegree)
							maskVec := utils.ExtractAndRepeatDiagonal(maskMat, rowIdx, modelParams.NumBatch)
							maskVecRot := utils.RotateSliceNew(maskVec, -i*baseLen*babyStep)
							localInnerEval.MulRelin(ctExp, maskVecRot, ctExp)
							localInnerEval.Rescale(ctExp, ctExp)
							localCtExpStore[j] = ctExp
							localInnerEval.Add(localCtAddGiantStepLocals[tInner], ctExp, localCtAddGiantStepLocals[tInner])
						}
					}(tInner, jStart, jEnd)
				}
				innerWg.Wait()

				// 合并localCtAddGiantStepLocals到localCtAddGiantStep
				for tInner := 0; tInner < numThreads; tInner++ {
					localEval.Add(localCtAddGiantStep, localCtAddGiantStepLocals[tInner], localCtAddGiantStep)
				}

				// 写回全局 ctExpStore 和 ctAddAllbyRow
				mu.Lock()
				for j, ctExp := range localCtExpStore {
					rowIdx := i*babyStep + j
					if rowIdx >= ctRows {
						continue
					}
					ctExpStore[rowIdx] = ctExp
				}
				ctAddRotate, _ := localEval.RotateNew(localCtAddGiantStep, i*baseLen*babyStep)
				eval.Add(ctAddAllbyRow, ctAddRotate, ctAddAllbyRow)
				mu.Unlock()
			}
		}(start, end)
	}
	wg.Wait()

	// Phase 4: 计算ctQKTInvSum
	// 预分配结果存储
	newCiphertexts := make([]*rlwe.Ciphertext, ctCols)
	for k := 0; k < ctCols; k++ {
		newCiphertexts[k] = ckks.NewCiphertext(*ckksParams, ctApproxMax.Degree(), ctApproxMax.Level())
	}

	// 单独线程计算 ctQKTInvSum
	var ctQKTInvSum *rlwe.Ciphertext
	var btsErr error
	var btsDone sync.WaitGroup
	btsDone.Add(1)
	go func() {
		defer btsDone.Done()
		localEval := eval.ShallowCopy()
		ctSumBts, err := btpEval.Bootstrap(ctAddAllbyRow)
		if err != nil {
			btsErr = err
			return
		}
		ctQKTInvSum = softmax.CiphertextInverse(ctSumBts, ckksParams, localEval, modelParams.InvMinValue, modelParams.InvMaxValue, modelParams.InvDegree, modelParams.InvIter)
	}()

	// BSGS 主体并行 (numThreads-1个线程)
	chunkSize = (giantStep + (numThreads - 2)) / (numThreads - 1) // 预留1个线程给Bootstrap
	for t := 0; t < numThreads-1; t++ {
		start := t * chunkSize
		end := min((t+1)*chunkSize, giantStep)
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			localEval := eval.ShallowCopy()
			localNewCiphertexts := make([]*rlwe.Ciphertext, ctCols)
			for k := 0; k < ctCols; k++ {
				localNewCiphertexts[k] = ckks.NewCiphertext(*ckksParams, ctApproxMax.Degree(), ctApproxMax.Level())
			}

			for i := start; i < end; i++ {
				ctQKV := &he.CiphertextMatrices{
					Ciphertexts: localNewCiphertexts,
					NumBatch:    ctBatch,
					NumRow:      ctRows,
					NumCol:      ctCols,
				}
				for j := 0; j < babyStep; j++ {
					rowIdx := i*babyStep + j
					if rowIdx >= ctRows {
						break
					}
					ctExp := ctExpStore[rowIdx]
					matrix.CiphertextMatricesMultiplyCiphertextAddToRes(ctVRotated[j], ctExp, ctQKV, ckksParams, localEval)
				}
				QKVRotKi := matrix.RotateCiphertextMatrices(ctQKV, i*baseLen*babyStep, localEval)
				// 合并到 newCiphertexts (线程安全)
				for k := 0; k < ctCols; k++ {
					// 这里 newCiphertexts[k] 是唯一写，按 index 分配无冲突
					localEval.Add(newCiphertexts[k], QKVRotKi.Ciphertexts[k], newCiphertexts[k])
				}
			}
		}(start, end)
	}
	wg.Wait()

	// 等 Bootstrap + Inverse 完成
	btsDone.Wait()
	if btsErr != nil {
		return nil, nil, btsErr
	}

	// phase 5: 计算ctQKTInvSum的乘积
	chunkSize = (ctCols + numThreads - 1) / numThreads
	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := (t + 1) * chunkSize
		if end > ctCols {
			end = ctCols
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			localEval := eval.ShallowCopy()
			for i := start; i < end; i++ {
				ct, err := localEval.MulRelinNew(newCiphertexts[i], ctQKTInvSum)
				if err != nil {
					panic(fmt.Sprintf("MulRelin error at index %d: %v", i, err))
				}
				localEval.Rescale(ct, ct)
				newCiphertexts[i] = ct // ← 线程安全：每个i只被唯一线程写
			}
		}(start, end)
	}
	wg.Wait()

	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ctBatch,
		NumRow:      ctRows,
		NumCol:      ctCols,
	}, ctQKTInvSum, nil
}

// ─────────────────────────────────────────────────────────────
// ①  外层：按 Head 并行
// ─────────────────────────────────────────────────────────────
func ComputeMultiHeadAttentionMT2(
	ctQ, ctK, ctV *he.CiphertextMatrices,
	modelParams *configs.ModelParams,
	ckksParams *ckks.Parameters,
	ecd *ckks.Encoder,
	enc *rlwe.Encryptor,
	eval *ckks.Evaluator,
	// btpParams *bootstrapping.Bootstrapper,
	btpEval *bootstrapping.Evaluator,
	numThreads int,
) (*he.CiphertextMatrices, error) {

	// 1) 拆头
	ctQsplit := matrix.SplitCiphertextMatricesByHeads(ctQ, modelParams)
	ctKsplit := matrix.SplitCiphertextMatricesByHeads(ctK, modelParams)
	ctVsplit := matrix.SplitCiphertextMatricesByHeads(ctV, modelParams)

	ctAttentionHeads := make([]*he.CiphertextMatrices, modelParams.NumHeads)

	var wg sync.WaitGroup
	var mu sync.Mutex
	var err error

	for headIdx := 0; headIdx < modelParams.NumHeads; headIdx++ {
		wg.Add(1)
		go func(h int) {
			defer wg.Done()

			// ★ 每个 head 建独立 Evaluator

			// startCopy := time.Now()
			localEval := eval.ShallowCopy()
			localEnc := enc.ShallowCopy()
			localEcd := ecd.ShallowCopy()
			// localBtpEval := btpEval.ShallowCopy()
			localBtpEval := btp.DeepCopyBootstrapEvaluator(btpEval, btpEval.Parameters)
			ctQHead := ctQsplit[h].CopyNew()
			ctKHead := ctKsplit[h].CopyNew()
			ctVHead := ctVsplit[h].CopyNew()
			// elapsedCopy := time.Since(startCopy)
			// fmt.Printf("[Head %d] 复制对象耗时: %v\n", h, elapsedCopy)

			// att, _, err := ComputeAttentionMT2(
			// 	ctQsplit[h], ctKsplit[h], ctVsplit[h],
			// 	modelParams, ckksParams, ecd, enc,
			// 	localEval, btpEval,
			// 	numThreads/modelParams.NumHeads)

			att, _, err := ComputeAttentionMT1_WorkerPool(
				ctQHead, ctKHead, ctVHead,
				modelParams, ckksParams, localEcd, localEnc,
				localEval, localBtpEval, numThreads/modelParams.NumHeads)
			// att, _, err := ComputeAttentionMT2(
			// 	ctQsplit[h], ctKsplit[h], ctVsplit[h],
			// 	modelParams, ckksParams, localEcd, localEnc,
			// 	localEval, localBtpEval, numThreads/modelParams.NumHeads)
			if err != nil {
				panic(err)
			}
			mu.Lock()
			ctAttentionHeads[h] = att
			mu.Unlock()
		}(headIdx)
	}
	wg.Wait()

	// 2) 合并头
	return matrix.MergeCiphertextMatricesByHeads(ctAttentionHeads), err
}

var maxGoroutines int32
var mu sync.Mutex

func updateMaxGoroutines() {
	mu.Lock()
	defer mu.Unlock()
	curr := int32(runtime.NumGoroutine())
	if curr > maxGoroutines {
		maxGoroutines = curr
	}
}

func ComputeAttentionMT2(
	ctQ, ctK, ctV *he.CiphertextMatrices,
	modelParams *configs.ModelParams,
	ckksParams *ckks.Parameters,
	ecd *ckks.Encoder,
	enc *rlwe.Encryptor,
	eval *ckks.Evaluator,
	btpEval *bootstrapping.Evaluator,
	numThreads int,
) (*he.CiphertextMatrices, *rlwe.Ciphertext, error) {
	ctRows, ctCols, ctBatch := ctQ.NumRow, ctQ.NumCol, ctQ.NumBatch
	babyStep, giantStep := modelParams.BabyStep, modelParams.GiantStep
	baseLen, realRow := modelParams.NumBatch, modelParams.NumRealRow

	// Step0. Consistency Check
	if ctQ.NumBatch != ctK.NumBatch || ctQ.NumBatch != ctV.NumBatch ||
		ctQ.NumRow != ctK.NumRow || ctQ.NumRow != ctV.NumRow ||
		ctQ.NumCol != ctK.NumCol || ctQ.NumCol != ctV.NumCol {
		return nil, nil, fmt.Errorf("ctQ, ctK, ctV shape mismatch")
	}

	// Generate Mask Matrix
	maskMat := utils.GenerateSpecialSquareMatrix(modelParams)

	// Generate Rotation Steps
	babySteps, giantSteps := make([]int, 0, babyStep), make([]int, 0, giantStep)
	for i := 0; i < babyStep; i++ {
		babySteps = append(babySteps, i*baseLen)
	}
	for i := 0; i < giantStep; i++ {
		giantSteps = append(giantSteps, -i*baseLen*babyStep)
	}

	// Batch Rotate Q, K, V
	ctQRotated := matrix.RotateCiphertextMatricesHoisting(ctQ, giantSteps, eval)
	ctKRotated := matrix.RotateCiphertextMatricesHoisting(ctK, babySteps, eval)
	ctVRotated := matrix.RotateCiphertextMatricesHoisting(ctV, babySteps, eval)

	// Step1. Compute QK^T via BSGS
	ctQKTCiphertext := make([]*rlwe.Ciphertext, ctRows)
	ctQKTAdd := ckks.NewCiphertext(*ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
	for i := 0; i < giantStep; i++ {
		ctAddGaintStep := ckks.NewCiphertext(*ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
		for j := 0; j < babyStep; j++ {
			if i*babyStep+j >= ctRows {
				break
			}
			ct, err := matrix.CiphertextMatricesMultiplyCiphertextMatricesThenAdd(ctQRotated[i], ctKRotated[j], ckksParams, eval)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to multiply ctQRotated[%d] and ctKRotated[%d]: %w", i, j, err)
			}
			eval.Add(ctAddGaintStep, ct, ctAddGaintStep)
			ctQKTCiphertext[i*babyStep+j] = ct
		}
		ctAddRotate, err := eval.RotateNew(ctAddGaintStep, i*baseLen*babyStep)
		if err != nil {
			panic(err)
		}
		eval.Add(ctQKTAdd, ctAddRotate, ctQKTAdd)
	}

	// Step2. Approximate Max using Mean and Std Estimation
	ctMean, err := eval.MulRelinNew(ctQKTAdd, 1/float64(realRow))
	if err != nil {
		panic(err)
	}
	eval.Rescale(ctMean, ctMean)
	ctQKTAddRot, err := eval.RotateHoistedNew(ctQKTAdd, giantSteps)
	if err != nil {
		panic(err)
	}

	constantValue := 2.08
	varScale := math.Sqrt(0.1*(1.0/float64(realRow))*constantValue) * (1.0 / float64(realRow))
	constValue := 1.25 * constantValue

	ctVarScale := ckks.NewCiphertext(*ckksParams, ctMean.Degree(), ctMean.Level())
	for i := 0; i < giantStep; i++ {
		ctAddGaintStep := ckks.NewCiphertext(*ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
		for j := 0; j < babyStep; j++ {
			if i*babyStep+j >= ctRows {
				break
			}
			ctQKTScale, err := eval.MulNew(ctQKTCiphertext[i*babyStep+j], realRow)
			if err != nil {
				panic(err)
			}
			ctSub, err := eval.SubNew(ctQKTScale, ctQKTAddRot[-i*baseLen*babyStep])
			if err != nil {
				panic(err)
			}

			maskVec := utils.ExtractAndRepeatDiagonal(maskMat, i*babyStep+j, modelParams.NumBatch)
			for idx := range maskVec {
				maskVec[idx] *= varScale
			}
			maskVecRot := utils.RotateSliceNew(maskVec, -i*baseLen*babyStep)

			eval.MulRelin(ctSub, maskVecRot, ctSub)
			eval.Rescale(ctSub, ctSub)
			eval.MulRelin(ctSub, ctSub, ctSub)
			eval.Rescale(ctSub, ctSub)
			eval.Add(ctAddGaintStep, ctSub, ctAddGaintStep)
		}
		ctAddRotate, err := eval.RotateNew(ctAddGaintStep, i*baseLen*babyStep)
		if err != nil {
			panic(err)
		}
		eval.Add(ctVarScale, ctAddRotate, ctVarScale)
	}

	ctApproxStd, err := eval.AddNew(ctVarScale, constValue)
	if err != nil {
		panic(err)
	}
	ctApproxMax, err := eval.AddNew(ctApproxStd, ctMean)
	if err != nil {
		panic(err)
	}
	ctApproxMaxRot, err := eval.RotateHoistedNew(ctApproxMax, giantSteps)
	if err != nil {
		panic(err)
	}

	// Step3. Compute Exp(QK - Max), Attention Weighting and QKV
	ctExpQKMinusMax := make([]*rlwe.Ciphertext, ctRows)
	ctAddAllbyRow := ckks.NewCiphertext(*ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
	for i := 0; i < giantStep; i++ {
		ctAddGaintStep := ckks.NewCiphertext(*ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
		for j := 0; j < babyStep; j++ {
			if i*babyStep+j >= ctRows {
				break
			}
			// Compute Exp(QK - Max)
			ctSub, err := eval.SubNew(ctQKTCiphertext[i*babyStep+j], ctApproxMaxRot[-i*baseLen*babyStep])
			if err != nil {
				panic(err)
			}
			ctExp := softmax.CiphertextExpChebyshev(ctSub, ckksParams, eval, modelParams.ExpMinValue, modelParams.ExpMaxValue, modelParams.ExpDegree)

			// Apply Mask
			maskVec := utils.ExtractAndRepeatDiagonal(maskMat, i*babyStep+j, modelParams.NumBatch)
			maskVecRot := utils.RotateSliceNew(maskVec, -i*baseLen*babyStep)
			eval.MulRelin(ctExp, maskVecRot, ctExp)
			eval.Rescale(ctExp, ctExp)

			ctExpQKMinusMax[i*babyStep+j] = ctExp // Cache Exp(QK - Max)

			// Sum for Softmax Denominator
			eval.Add(ctAddGaintStep, ctExp, ctAddGaintStep)
		}
		ctAddRotate, err := eval.RotateNew(ctAddGaintStep, i*baseLen*babyStep)
		if err != nil {
			panic(err)
		}
		eval.Add(ctAddAllbyRow, ctAddRotate, ctAddAllbyRow)
	}

	// Step2. Bootstrap Softmax Sum (Denominator)
	ctSumBts, err := btpEval.Bootstrap(ctAddAllbyRow)
	if err != nil {
		panic(err)
	}

	// Step3. Compute Inverse Sum (Softmax Normalization)
	ctQKTInvSum := softmax.CiphertextInverse(ctSumBts, ckksParams, eval, modelParams.InvMinValue, modelParams.InvMaxValue, modelParams.InvDegree, modelParams.InvIter)

	// Step5. Compute Attention * V via BSGS
	newCiphertexts := make([]*rlwe.Ciphertext, ctCols)
	for k := 0; k < ctCols; k++ {
		newCiphertexts[k] = ckks.NewCiphertext(*ckksParams, ctQKTCiphertext[0].Degree(), ctQKTCiphertext[0].Level())
	}
	for i := 0; i < giantStep; i++ {
		localNewCiphertexts := make([]*rlwe.Ciphertext, ctV.NumCol)
		for k := 0; k < ctV.NumCol; k++ {
			localNewCiphertexts[k] = ckks.NewCiphertext(*ckksParams, ctQKTCiphertext[0].Degree(), ctQKTCiphertext[0].Level())
		}
		ctQKV := &he.CiphertextMatrices{
			Ciphertexts: localNewCiphertexts,
			NumBatch:    ctBatch,
			NumRow:      ctRows,
			NumCol:      ctCols,
		}
		for j := 0; j < babyStep; j++ {
			if i*babyStep+j >= ctRows {
				break
			}
			// Multiply Softmax Weight * V
			err := matrix.CiphertextMatricesMultiplyCiphertextAddToRes(ctVRotated[j], ctExpQKMinusMax[i*babyStep+j], ctQKV, ckksParams, eval)
			if err != nil {
				panic(err)
			}
		}
		QKVRotKi := matrix.RotateCiphertextMatrices(ctQKV, i*baseLen*babyStep, eval)
		for k := 0; k < ctCols; k++ {
			eval.Add(newCiphertexts[k], QKVRotKi.Ciphertexts[k], newCiphertexts[k])
		}
	}
	// Step6. Multiply QKV by Softmax Normalization
	finalCiphertexts := make([]*rlwe.Ciphertext, ctCols)
	for i := 0; i < ctCols; i++ {
		ct, err := eval.MulRelinNew(newCiphertexts[i], ctQKTInvSum)
		if err != nil {
			panic("failed to multiply QKV with inverse sum: " + err.Error())
		}
		eval.Rescale(ct, ct)
		finalCiphertexts[i] = ct
	}

	return &he.CiphertextMatrices{
		Ciphertexts: finalCiphertexts,
		NumRow:      ctRows,
		NumCol:      ctCols,
		NumBatch:    ctBatch,
	}, ctSumBts, nil
}

func ComputeAttentionMerged(
	ctQ, ctK, ctV *he.CiphertextMatrices,
	modelParams *configs.ModelParams,
	ckksParams *ckks.Parameters,
	ecd *ckks.Encoder,
	enc *rlwe.Encryptor,
	eval *ckks.Evaluator,
	btpEval *bootstrapping.Evaluator,
	numThreads int,
) (*he.CiphertextMatrices, *rlwe.Ciphertext, error) {
	ctRows, ctCols, ctBatch := ctQ.NumRow, ctQ.NumCol, ctQ.NumBatch
	babyStep, giantStep := modelParams.BabyStep, modelParams.GiantStep
	baseLen, realRow := modelParams.NumBatch, modelParams.NumRealRow

	// Step0. Consistency Check
	if ctQ.NumBatch != ctK.NumBatch || ctQ.NumBatch != ctV.NumBatch ||
		ctQ.NumRow != ctK.NumRow || ctQ.NumRow != ctV.NumRow ||
		ctQ.NumCol != ctK.NumCol || ctQ.NumCol != ctV.NumCol {
		return nil, nil, fmt.Errorf("ctQ, ctK, ctV shape mismatch")
	}

	// Generate Mask Matrix
	maskMat := utils.GenerateSpecialSquareMatrix(modelParams)

	// Generate Rotation Steps
	babySteps, giantSteps := make([]int, 0, babyStep), make([]int, 0, giantStep)
	for i := 0; i < babyStep; i++ {
		babySteps = append(babySteps, i*baseLen)
	}
	for i := 0; i < giantStep; i++ {
		giantSteps = append(giantSteps, -i*baseLen*babyStep)
	}

	// Batch Rotate Q, K, V
	ctQRotated := matrix.RotateCiphertextMatricesHoisting(ctQ, giantSteps, eval)
	ctKRotated := matrix.RotateCiphertextMatricesHoisting(ctK, babySteps, eval)
	ctVRotated := matrix.RotateCiphertextMatricesHoisting(ctV, babySteps, eval)

	// Step1. Compute QK^T via BSGS
	ctQKTCiphertext := make([]*rlwe.Ciphertext, ctRows)
	ctQKTAdd := ckks.NewCiphertext(*ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
	for i := 0; i < giantStep; i++ {
		ctAddGaintStep := ckks.NewCiphertext(*ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
		for j := 0; j < babyStep; j++ {
			if i*babyStep+j >= ctRows {
				break
			}
			ct, err := matrix.CiphertextMatricesMultiplyCiphertextMatricesThenAdd(ctQRotated[i], ctKRotated[j], ckksParams, eval)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to multiply ctQRotated[%d] and ctKRotated[%d]: %w", i, j, err)
			}
			eval.Add(ctAddGaintStep, ct, ctAddGaintStep)
			ctQKTCiphertext[i*babyStep+j] = ct
		}
		ctAddRotate, err := eval.RotateNew(ctAddGaintStep, i*baseLen*babyStep)
		if err != nil {
			panic(err)
		}
		eval.Add(ctQKTAdd, ctAddRotate, ctQKTAdd)
	}

	// Step2. Approximate Max using Mean and Std Estimation
	ctMean, err := eval.MulRelinNew(ctQKTAdd, 1/float64(realRow))
	if err != nil {
		panic(err)
	}
	eval.Rescale(ctMean, ctMean)
	ctQKTAddRot, err := eval.RotateHoistedNew(ctQKTAdd, giantSteps)
	if err != nil {
		panic(err)
	}

	constantValue := 2.08
	varScale := math.Sqrt(0.1*(1.0/float64(realRow))*constantValue) * (1.0 / float64(realRow))
	constValue := 1.25 * constantValue

	ctVarScale := ckks.NewCiphertext(*ckksParams, ctMean.Degree(), ctMean.Level())
	for i := 0; i < giantStep; i++ {
		ctAddGaintStep := ckks.NewCiphertext(*ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
		for j := 0; j < babyStep; j++ {
			if i*babyStep+j >= ctRows {
				break
			}
			ctQKTScale, err := eval.MulNew(ctQKTCiphertext[i*babyStep+j], realRow)
			if err != nil {
				panic(err)
			}
			ctSub, err := eval.SubNew(ctQKTScale, ctQKTAddRot[-i*baseLen*babyStep])
			if err != nil {
				panic(err)
			}

			maskVec := utils.ExtractAndRepeatDiagonal(maskMat, i*babyStep+j, modelParams.NumBatch)
			for idx := range maskVec {
				maskVec[idx] *= varScale
			}
			maskVecRot := utils.RotateSliceNew(maskVec, -i*baseLen*babyStep)

			eval.MulRelin(ctSub, maskVecRot, ctSub)
			eval.Rescale(ctSub, ctSub)
			eval.MulRelin(ctSub, ctSub, ctSub)
			eval.Rescale(ctSub, ctSub)
			eval.Add(ctAddGaintStep, ctSub, ctAddGaintStep)
		}
		ctAddRotate, err := eval.RotateNew(ctAddGaintStep, i*baseLen*babyStep)
		if err != nil {
			panic(err)
		}
		eval.Add(ctVarScale, ctAddRotate, ctVarScale)
	}

	ctApproxStd, err := eval.AddNew(ctVarScale, constValue)
	if err != nil {
		panic(err)
	}
	ctApproxMax, err := eval.AddNew(ctApproxStd, ctMean)
	if err != nil {
		panic(err)
	}
	ctApproxMaxRot, err := eval.RotateHoistedNew(ctApproxMax, giantSteps)
	if err != nil {
		panic(err)
	}

	// Step3. Compute Exp(QK - Max), Attention Weighting and QKV
	ctExpQKMinusMax := make([]*rlwe.Ciphertext, ctRows)
	ctAddAllbyRow := ckks.NewCiphertext(*ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
	for i := 0; i < giantStep; i++ {
		ctAddGaintStep := ckks.NewCiphertext(*ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
		for j := 0; j < babyStep; j++ {
			if i*babyStep+j >= ctRows {
				break
			}
			// Compute Exp(QK - Max)
			ctSub, err := eval.SubNew(ctQKTCiphertext[i*babyStep+j], ctApproxMaxRot[-i*baseLen*babyStep])
			if err != nil {
				panic(err)
			}
			ctExp := softmax.CiphertextExpChebyshev(ctSub, ckksParams, eval, modelParams.ExpMinValue, modelParams.ExpMaxValue, modelParams.ExpDegree)

			// Apply Mask
			maskVec := utils.ExtractAndRepeatDiagonal(maskMat, i*babyStep+j, modelParams.NumBatch)
			maskVecRot := utils.RotateSliceNew(maskVec, -i*baseLen*babyStep)
			eval.MulRelin(ctExp, maskVecRot, ctExp)
			eval.Rescale(ctExp, ctExp)

			ctExpQKMinusMax[i*babyStep+j] = ctExp // Cache Exp(QK - Max)

			// Sum for Softmax Denominator
			eval.Add(ctAddGaintStep, ctExp, ctAddGaintStep)
		}
		ctAddRotate, err := eval.RotateNew(ctAddGaintStep, i*baseLen*babyStep)
		if err != nil {
			panic(err)
		}
		eval.Add(ctAddAllbyRow, ctAddRotate, ctAddAllbyRow)
	}

	// Step2. Bootstrap Softmax Sum (Denominator)
	ctSumBts, err := btpEval.Bootstrap(ctAddAllbyRow)
	if err != nil {
		panic(err)
	}

	// Step3. Compute Inverse Sum (Softmax Normalization)
	ctQKTInvSum := softmax.CiphertextInverse(ctSumBts, ckksParams, eval, modelParams.InvMinValue, modelParams.InvMaxValue, modelParams.InvDegree, modelParams.InvIter)

	// Step5. Compute Attention * V via BSGS
	newCiphertexts := make([]*rlwe.Ciphertext, ctCols)
	for k := 0; k < ctCols; k++ {
		newCiphertexts[k] = ckks.NewCiphertext(*ckksParams, ctQKTCiphertext[0].Degree(), ctQKTCiphertext[0].Level())
	}
	for i := 0; i < giantStep; i++ {
		localNewCiphertexts := make([]*rlwe.Ciphertext, ctV.NumCol)
		for k := 0; k < ctV.NumCol; k++ {
			localNewCiphertexts[k] = ckks.NewCiphertext(*ckksParams, ctQKTCiphertext[0].Degree(), ctQKTCiphertext[0].Level())
		}
		ctQKV := &he.CiphertextMatrices{
			Ciphertexts: localNewCiphertexts,
			NumBatch:    ctBatch,
			NumRow:      ctRows,
			NumCol:      ctCols,
		}
		for j := 0; j < babyStep; j++ {
			if i*babyStep+j >= ctRows {
				break
			}
			// Multiply Softmax Weight * V
			err := matrix.CiphertextMatricesMultiplyCiphertextAddToRes(ctVRotated[j], ctExpQKMinusMax[i*babyStep+j], ctQKV, ckksParams, eval)
			if err != nil {
				panic(err)
			}
		}
		QKVRotKi := matrix.RotateCiphertextMatrices(ctQKV, i*baseLen*babyStep, eval)
		for k := 0; k < ctCols; k++ {
			eval.Add(newCiphertexts[k], QKVRotKi.Ciphertexts[k], newCiphertexts[k])
		}
	}
	// Step6. Multiply QKV by Softmax Normalization
	finalCiphertexts := make([]*rlwe.Ciphertext, ctCols)
	for i := 0; i < ctCols; i++ {
		ct, err := eval.MulRelinNew(newCiphertexts[i], ctQKTInvSum)
		if err != nil {
			panic("failed to multiply QKV with inverse sum: " + err.Error())
		}
		eval.Rescale(ct, ct)
		finalCiphertexts[i] = ct
	}

	return &he.CiphertextMatrices{
		Ciphertexts: finalCiphertexts,
		NumRow:      ctRows,
		NumCol:      ctCols,
		NumBatch:    ctBatch,
	}, ctSumBts, nil
}

func ComputeAttentionMT3(
	ctQ, ctK, ctV *he.CiphertextMatrices,
	modelParams *configs.ModelParams,
	ckksParams *ckks.Parameters,
	ecd *ckks.Encoder,
	enc *rlwe.Encryptor,
	eval *ckks.Evaluator,
	btpEval *bootstrapping.Evaluator,
	numThreads int,
) (*he.CiphertextMatrices, *rlwe.Ciphertext, error) {
	ctRows, ctCols, ctBatch := ctQ.NumRow, ctQ.NumCol, ctQ.NumBatch
	babyStep, giantStep := modelParams.BabyStep, modelParams.GiantStep
	baseLen, realRow := modelParams.NumBatch, modelParams.NumRealRow

	// Step0. Consistency Check
	if ctQ.NumBatch != ctK.NumBatch || ctQ.NumBatch != ctV.NumBatch ||
		ctQ.NumRow != ctK.NumRow || ctQ.NumRow != ctV.NumRow ||
		ctQ.NumCol != ctK.NumCol || ctQ.NumCol != ctV.NumCol {
		return nil, nil, fmt.Errorf("ctQ, ctK, ctV shape mismatch")
	}

	// Generate Mask Matrix
	maskMat := utils.GenerateSpecialSquareMatrix(modelParams)

	// Generate Rotation Steps
	babySteps, giantSteps := make([]int, 0, babyStep), make([]int, 0, giantStep)
	for i := 0; i < babyStep; i++ {
		babySteps = append(babySteps, i*baseLen)
	}
	for i := 0; i < giantStep; i++ {
		giantSteps = append(giantSteps, -i*baseLen*babyStep)
	}

	// Batch Rotate Q, K, V
	ctQRotated := matrix.RotateCiphertextMatricesHoistingMT(ctQ, giantSteps, eval, numThreads)
	ctKRotated := matrix.RotateCiphertextMatricesHoistingMT(ctK, babySteps, eval, numThreads)
	ctVRotated := matrix.RotateCiphertextMatricesHoistingMT(ctV, babySteps, eval, numThreads)

	// Step1. Compute QK^T via BSGS
	ctQKTCiphertext := make([]*rlwe.Ciphertext, ctRows)
	// 不要用 NewCiphertext 直接做全局累加器，首项用 CopyNew 对齐 scale/level
	var ctQKTAdd *rlwe.Ciphertext
	var muQKTAdd sync.Mutex

	// 并发预算（外层×内层 ≈ numThreads）
	outerThreads := numThreads
	if outerThreads > giantStep {
		outerThreads = giantStep
	}
	if outerThreads < 1 {
		outerThreads = 1
	}
	outerChunk := (giantStep + outerThreads - 1) / outerThreads

	var wg sync.WaitGroup
	for o := 0; o < outerThreads; o++ {
		startI := o * outerChunk
		endI := (o + 1) * outerChunk
		if endI > giantStep {
			endI = giantStep
		}
		if startI >= endI {
			continue
		}
		wg.Add(1)
		go func(startI, endI int) {
			defer wg.Done()
			localEval := eval.ShallowCopy()

			// 本 goroutine 的“旋转后局部总和”
			var localRotSum *rlwe.Ciphertext

			for i := startI; i < endI; i++ {
				// —— 内层并行切 j 段
				// 内层线程数按预算分配，且不超过 babyStep
				innerThreads := numThreads / outerThreads
				if innerThreads < 1 {
					innerThreads = 1
				}
				if innerThreads > babyStep {
					innerThreads = babyStep
				}
				innerChunk := (babyStep + innerThreads - 1) / innerThreads

				innerSums := make([]*rlwe.Ciphertext, innerThreads)
				var wgInner sync.WaitGroup
				for it := 0; it < innerThreads; it++ {
					jStart := it * innerChunk
					jEnd := (it + 1) * innerChunk
					if jEnd > babyStep {
						jEnd = babyStep
					}
					if jStart >= jEnd {
						continue
					}
					wgInner.Add(1)
					go func(slot, jStart, jEnd int) {
						defer wgInner.Done()
						le := localEval.ShallowCopy() // 内层各自 evaluator
						var part *rlwe.Ciphertext     // 该段 j 的局部和（未旋转）

						for j := jStart; j < jEnd; j++ {
							rowIdx := i*babyStep + j
							if rowIdx >= ctRows {
								break
							}
							ct, err := matrix.CiphertextMatricesMultiplyCiphertextMatricesThenAdd(
								ctQRotated[i], ctKRotated[j], ckksParams, le)
							if err != nil {
								panic(err)
							}
							if part == nil {
								part = ct.CopyNew() // ★ 首项 CopyNew，保证 scale/level 对齐
							} else {
								le.Add(part, ct, part)
							}
							// 每个 rowIdx 仅此处写一次 → 线程安全
							ctQKTCiphertext[rowIdx] = ct
						}
						innerSums[slot] = part
					}(it, jStart, jEnd)
				}
				wgInner.Wait()

				// 把所有内层段的和，合并成该 i 的“未旋转总和”
				var localStep *rlwe.Ciphertext
				for _, part := range innerSums {
					if part == nil {
						continue
					}
					if localStep == nil {
						localStep = part.CopyNew()
					} else {
						localEval.Add(localStep, part, localStep)
					}
				}
				if localStep == nil {
					continue
				}

				// 旋转对齐该 giantStep
				rot, err := localEval.RotateNew(localStep, i*baseLen*babyStep)
				if err != nil {
					panic(err)
				}
				// 累到本 goroutine 的“旋转后局部总和”
				if localRotSum == nil {
					localRotSum = rot.CopyNew()
				} else {
					localEval.Add(localRotSum, rot, localRotSum)
				}
			}

			// 把本 goroutine 的“旋转后局部总和”一次性合并到全局 ctQKTAdd
			if localRotSum != nil {
				muQKTAdd.Lock()
				if ctQKTAdd == nil {
					ctQKTAdd = localRotSum.CopyNew()
				} else {
					// 用同一个 eval 合并更稳（避免不同 evaluator 的内部状态差异）
					eval.Add(ctQKTAdd, localRotSum, ctQKTAdd)
				}
				muQKTAdd.Unlock()
			}
		}(startI, endI)
	}
	wg.Wait()

	// ctScale := &ctQKTAdd.Scale.Value // We need to access the pointer in order for it to display correctly in the command line.
	// fmt.Printf("ctQKTAdd Scale rescaling: %f\n", ctScale)

	// Step2. Approximate Max using Mean and Std Estimation (并行版)
	var (
		ctMean      *rlwe.Ciphertext
		ctQKTAddRot map[int]*rlwe.Ciphertext
		errMean     error
		errRot      error
	)

	// var wg sync.WaitGroup
	wg.Add(2)

	// 线程1：计算均值
	go func() {
		defer wg.Done()
		localEval := eval.ShallowCopy()
		ctMean, errMean = localEval.MulRelinNew(ctQKTAdd, 1/float64(realRow))
		if errMean != nil {
			return
		}
		localEval.Rescale(ctMean, ctMean)
	}()

	// 线程2：一次性做 hoisted 旋转
	go func() {
		defer wg.Done()
		localEval := eval.ShallowCopy()
		ctQKTAddRot, errRot = localEval.RotateHoistedNew(ctQKTAdd, giantSteps)
	}()

	wg.Wait()
	if errMean != nil {
		panic(errMean)
	}
	if errRot != nil {
		panic(errRot)
	}

	// ctScale = &ctMean.Scale.Value // We need to access the pointer in order for it to display correctly in the command line.
	// fmt.Printf("ctMean Scale rescaling: %f\n", ctScale)

	constantValue := modelParams.ConstantValue
	varScale := math.Sqrt(0.1*(1.0/float64(realRow))*constantValue) * (1.0 / float64(realRow))
	constValue := 1.25 * constantValue

	// ⚠️ 改：不要先 New 一个空累加器，避免与首项 scale/level 不对齐
	var ctVarScale *rlwe.Ciphertext
	var muVarScale sync.Mutex

	// // 并发预算
	// outerThreads := numThreads
	// if outerThreads > giantStep {
	// 	outerThreads = giantStep
	// }
	// if outerThreads < 1 {
	// 	outerThreads = 1
	// }
	// outerChunk := (giantStep + outerThreads - 1) / outerThreads

	// var wg sync.WaitGroup
	for o := 0; o < outerThreads; o++ {
		startI := o * outerChunk
		endI := (o + 1) * outerChunk
		if endI > giantStep {
			endI = giantStep
		}
		if startI >= endI {
			continue
		}

		wg.Add(1)
		go func(startI, endI int) {
			defer wg.Done()
			localEval := eval.ShallowCopy()

			// 本 goroutine 的“旋转后局部总和”
			var localRotSum *rlwe.Ciphertext

			for i := startI; i < endI; i++ {
				rotKey := -i * baseLen * babyStep
				ref, ok := ctQKTAddRot[rotKey]
				if !ok || ref == nil {
					panic(fmt.Errorf("missing hoisted rotation for step %d", rotKey))
				}

				// 内层 j 再并行
				innerThreads := numThreads / outerThreads
				if innerThreads < 1 {
					innerThreads = 1
				}
				if innerThreads > babyStep {
					innerThreads = babyStep
				}
				innerChunk := (babyStep + innerThreads - 1) / innerThreads

				partials := make([]*rlwe.Ciphertext, innerThreads)
				var iwg sync.WaitGroup
				for it := 0; it < innerThreads; it++ {
					jStart := it * innerChunk
					jEnd := (it + 1) * innerChunk
					if jEnd > babyStep {
						jEnd = babyStep
					}
					if jStart >= jEnd {
						continue
					}

					iwg.Add(1)
					go func(slot, jStart, jEnd int) {
						defer iwg.Done()
						le := localEval.ShallowCopy()

						var part *rlwe.Ciphertext // 该段 j 的局部和（未旋转）
						for j := jStart; j < jEnd; j++ {
							rowIdx := i*babyStep + j
							if rowIdx >= ctRows {
								break
							}

							ctQKTScale, err := le.MulNew(ctQKTCiphertext[rowIdx], realRow)
							if err != nil {
								panic(err)
							}
							ctSub, err := le.SubNew(ctQKTScale, ref)
							if err != nil {
								panic(err)
							}

							maskVec := utils.ExtractAndRepeatDiagonal(maskMat, rowIdx, modelParams.NumBatch)
							for idx := range maskVec {
								maskVec[idx] *= varScale
							}
							maskVecRot := utils.RotateSliceNew(maskVec, -i*baseLen*babyStep)

							le.MulRelin(ctSub, maskVecRot, ctSub)
							le.Rescale(ctSub, ctSub)
							le.MulRelin(ctSub, ctSub, ctSub)
							le.Rescale(ctSub, ctSub)

							if part == nil {
								part = ctSub.CopyNew() // ★ 首项用 CopyNew 保证 scale/level 对齐
							} else {
								le.Add(part, ctSub, part)
							}
						}
						partials[slot] = part
					}(it, jStart, jEnd)
				}
				iwg.Wait()

				// 合并内层段得到该 i 的“未旋转总和”
				var localStep *rlwe.Ciphertext
				for _, p := range partials {
					if p == nil {
						continue
					}
					if localStep == nil {
						localStep = p.CopyNew()
					} else {
						localEval.Add(localStep, p, localStep)
					}
				}
				if localStep == nil {
					continue
				}

				// 旋转对齐该 giantStep，再累到本 goroutine 的“旋转后局部总和”
				rot, err := localEval.RotateNew(localStep, i*baseLen*babyStep)
				if err != nil {
					panic(err)
				}
				if localRotSum == nil {
					localRotSum = rot.CopyNew()
				} else {
					localEval.Add(localRotSum, rot, localRotSum)
				}
			}

			// 合并到全局 ctVarScale（加锁；用同一 eval 做 Add 更稳）
			if localRotSum != nil {
				muVarScale.Lock()
				if ctVarScale == nil {
					ctVarScale = localRotSum.CopyNew()
				} else {
					eval.Add(ctVarScale, localRotSum, ctVarScale)
				}
				muVarScale.Unlock()
			}
		}(startI, endI)
	}
	wg.Wait()

	// 串行部分
	ctApproxStd, err := eval.AddNew(ctVarScale, constValue)
	if err != nil {
		panic(err)
	}
	ctApproxMax, err := eval.AddNew(ctApproxStd, ctMean)
	if err != nil {
		panic(err)
	}
	ctApproxMaxRot, err := eval.RotateHoistedNew(ctApproxMax, giantSteps)
	if err != nil {
		panic(err)
	}

	// ─────────────────────────────────────────────────────────────
	// Step3. Compute Exp(QK - Max), and accumulate denominator
	// 外层 i 并行 + 内层 j 并行（局部→旋转→局部和→全局合并）
	// ─────────────────────────────────────────────────────────────
	ctExpQKMinusMax := make([]*rlwe.Ciphertext, ctRows)

	// 不要预先 New；用首项 CopyNew 对齐 scale/level
	var ctAddAllbyRow *rlwe.Ciphertext
	var muAddAll sync.Mutex

	// // 并发预算
	// outerThreads := numThreads
	// if outerThreads > giantStep {
	// 	outerThreads = giantStep
	// }
	// if outerThreads < 1 {
	// 	outerThreads = 1
	// }
	// outerChunk := (giantStep + outerThreads - 1) / outerThreads

	// var wg sync.WaitGroup
	for o := 0; o < outerThreads; o++ {
		startI := o * outerChunk
		endI := (o + 1) * outerChunk
		if endI > giantStep {
			endI = giantStep
		}
		if startI >= endI {
			continue
		}

		wg.Add(1)
		go func(startI, endI int) {
			defer wg.Done()
			localEval := eval.ShallowCopy()

			// 本 goroutine 的“旋转后局部总和”
			var localRotSum *rlwe.Ciphertext

			for i := startI; i < endI; i++ {
				rotKey := -i * baseLen * babyStep
				ref, ok := ctApproxMaxRot[rotKey]
				if !ok || ref == nil {
					panic(fmt.Errorf("missing hoisted rotation for step %d", rotKey))
				}

				// 内层 j 并行（把 babyStep 切段）
				innerThreads := numThreads / outerThreads
				if innerThreads < 1 {
					innerThreads = 1
				}
				if innerThreads > babyStep {
					innerThreads = babyStep
				}
				innerChunk := (babyStep + innerThreads - 1) / innerThreads

				partials := make([]*rlwe.Ciphertext, innerThreads)
				var iwg sync.WaitGroup
				for it := 0; it < innerThreads; it++ {
					jStart := it * innerChunk
					jEnd := (it + 1) * innerChunk
					if jEnd > babyStep {
						jEnd = babyStep
					}
					if jStart >= jEnd {
						continue
					}

					iwg.Add(1)
					go func(slot, jStart, jEnd int) {
						defer iwg.Done()
						le := localEval.ShallowCopy()
						var part *rlwe.Ciphertext // 该段 j 的局部（未旋转）和

						for j := jStart; j < jEnd; j++ {
							rowIdx := i*babyStep + j
							if rowIdx >= ctRows {
								break
							}

							// Exp(QK - Max)
							ctSub, err := le.SubNew(ctQKTCiphertext[rowIdx], ref)
							if err != nil {
								panic(err)
							}
							ctExp := softmax.CiphertextExpChebyshev(
								ctSub, ckksParams, le,
								modelParams.ExpMinValue, modelParams.ExpMaxValue, modelParams.ExpDegree,
							)

							// Mask
							maskVec := utils.ExtractAndRepeatDiagonal(maskMat, rowIdx, modelParams.NumBatch)
							maskVecRot := utils.RotateSliceNew(maskVec, -i*baseLen*babyStep)
							le.MulRelin(ctExp, maskVecRot, ctExp)
							le.Rescale(ctExp, ctExp)

							// 写回缓存（每个 rowIdx 只写一次 → 并发安全）
							ctExpQKMinusMax[rowIdx] = ctExp

							// 局部累加（首项 CopyNew）
							if part == nil {
								part = ctExp.CopyNew()
							} else {
								le.Add(part, ctExp, part)
							}
						}
						partials[slot] = part
					}(it, jStart, jEnd)
				}
				iwg.Wait()

				// 合并内层段 → 得到该 i 的未旋转和
				var localStep *rlwe.Ciphertext
				for _, p := range partials {
					if p == nil {
						continue
					}
					if localStep == nil {
						localStep = p.CopyNew()
					} else {
						localEval.Add(localStep, p, localStep)
					}
				}
				if localStep == nil {
					continue
				}

				// 旋转对齐该 giantStep，累到本 goroutine 的“旋转后局部总和”
				rot, err := localEval.RotateNew(localStep, i*baseLen*babyStep)
				if err != nil {
					panic(err)
				}
				if localRotSum == nil {
					localRotSum = rot.CopyNew()
				} else {
					localEval.Add(localRotSum, rot, localRotSum)
				}
			}

			// 合并到全局 ctAddAllbyRow（加锁；用同一 eval 合并更稳）
			if localRotSum != nil {
				muAddAll.Lock()
				if ctAddAllbyRow == nil {
					ctAddAllbyRow = localRotSum.CopyNew()
				} else {
					eval.Add(ctAddAllbyRow, localRotSum, ctAddAllbyRow)
				}
				muAddAll.Unlock()
			}
		}(startI, endI)
	}
	wg.Wait()

	// ─────────────────────────────────────────────────────────────
	// Step4: Bootstrap + Inverse   （单独 1 个线程）
	// Step5: (Exp * V) via BSGS    （其余线程并行）
	// ─────────────────────────────────────────────────────────────

	// 先准备全局累加槽（先置 nil，合并时首项用 CopyNew 对齐 scale/level）
	newCiphertexts := make([]*rlwe.Ciphertext, ctCols)

	// 1) 专线线程：Bootstrap + Inverse
	var (
		ctQKTInvSum *rlwe.Ciphertext
		btsErr      error
	)
	var btsWG sync.WaitGroup
	btsWG.Add(1)
	go func() {
		defer btsWG.Done()
		localEval := eval.ShallowCopy()
		ctSumBts, err := btpEval.Bootstrap(ctAddAllbyRow) // 只在这条专线里用 btpEval，避免竞态
		if err != nil {
			btsErr = err
			return
		}
		ctQKTInvSum = softmax.CiphertextInverse(
			ctSumBts, ckksParams, localEval,
			modelParams.InvMinValue, modelParams.InvMaxValue, modelParams.InvDegree, modelParams.InvIter)
	}()

	// 2) 其余线程：(Exp * V) 的 BSGS 并行累加
	workersForQKV := numThreads - 1
	if workersForQKV < 1 {
		workersForQKV = 1
	}
	if workersForQKV > giantStep {
		workersForQKV = giantStep
	}

	outerChunk = (giantStep + workersForQKV - 1) / workersForQKV
	// var wg sync.WaitGroup
	var muNew sync.Mutex

	for w := 0; w < workersForQKV; w++ {
		startI := w * outerChunk
		endI := (w + 1) * outerChunk
		if endI > giantStep {
			endI = giantStep
		}
		if startI >= endI {
			continue
		}

		wg.Add(1)
		go func(startI, endI int) {
			defer wg.Done()
			localEval := eval.ShallowCopy()
			// 本 goroutine 的列向量局部和
			localNew := make([]*rlwe.Ciphertext, ctCols)

			for i := startI; i < endI; i++ {
				// 为该 i 建一个本地矩阵累加器（与其它 goroutine 无共享）
				qkv := &he.CiphertextMatrices{
					Ciphertexts: make([]*rlwe.Ciphertext, ctCols),
					NumBatch:    ctBatch, NumRow: ctRows, NumCol: ctCols,
				}
				// 这里内层 j 如需再做二级并行，也按你前面“双层并行”的模式切分；
				// 先给出串行 j（通常已足够，且最稳）：
				for j := 0; j < babyStep; j++ {
					rowIdx := i*babyStep + j
					if rowIdx >= ctRows {
						break
					}
					// Multiply Softmax Weight * V
					if err := matrix.CiphertextMatricesMultiplyCiphertextAddToRes(
						ctVRotated[j], ctExpQKMinusMax[rowIdx], qkv, ckksParams, localEval); err != nil {
						panic(err)
					}
				}
				// 旋转到目标对齐
				qkvRot := matrix.RotateCiphertextMatrices(qkv, i*baseLen*babyStep, localEval)

				// 累到本 goroutine 的列向量局部和（首项用 CopyNew）
				for k := 0; k < ctCols; k++ {
					ct := qkvRot.Ciphertexts[k]
					if ct == nil {
						continue
					}
					if localNew[k] == nil {
						localNew[k] = ct.CopyNew()
					} else {
						localEval.Add(localNew[k], ct, localNew[k])
					}
				}
			}

			// 合并本 goroutine 的结果到全局（加锁；首项 CopyNew）
			muNew.Lock()
			for k := 0; k < ctCols; k++ {
				if localNew[k] == nil {
					continue
				}
				if newCiphertexts[k] == nil {
					newCiphertexts[k] = localNew[k].CopyNew()
				} else {
					eval.Add(newCiphertexts[k], localNew[k], newCiphertexts[k])
				}
			}
			muNew.Unlock()
		}(startI, endI)
	}
	wg.Wait()

	// 等专线 Bootstrap + Inverse 完成
	btsWG.Wait()
	if btsErr != nil {
		return nil, nil, btsErr
	}

	// ─────────────────────────────────────────────────────────────
	// Step6: 按列并行乘以 ctQKTInvSum（归一化）
	// 这里假定：如果前面 Bootstrap+Inverse 是独立 goroutine，已经 invWG.Wait()。
	// ─────────────────────────────────────────────────────────────

	finalCiphertexts := make([]*rlwe.Ciphertext, ctCols)

	if numThreads <= 1 || ctCols <= 1 {
		// 串行退化
		for i := 0; i < ctCols; i++ {
			if newCiphertexts[i] == nil {
				continue
			}
			ct, err := eval.MulRelinNew(newCiphertexts[i], ctQKTInvSum)
			if err != nil {
				panic(fmt.Errorf("MulRelin col=%d: %w", i, err))
			}
			eval.Rescale(ct, ct)
			finalCiphertexts[i] = ct
		}
	} else {
		// 并行：按列分块
		threads := numThreads
		if threads > ctCols {
			threads = ctCols
		}
		chunk := (ctCols + threads - 1) / threads

		var wg sync.WaitGroup
		for t := 0; t < threads; t++ {
			start := t * chunk
			end := (t + 1) * chunk
			if end > ctCols {
				end = ctCols
			}
			if start >= end {
				continue
			}

			wg.Add(1)
			go func(start, end int) {
				defer wg.Done()
				localEval := eval.ShallowCopy()
				for i := start; i < end; i++ {
					src := newCiphertexts[i]
					if src == nil {
						continue
					} // 允许稀疏
					ct, err := localEval.MulRelinNew(src, ctQKTInvSum)
					if err != nil {
						panic(fmt.Errorf("MulRelin col=%d: %w", i, err))
					}
					localEval.Rescale(ct, ct)
					finalCiphertexts[i] = ct // 每列唯一写，无需加锁
				}
			}(start, end)
		}
		wg.Wait()
	}

	return &he.CiphertextMatrices{
		Ciphertexts: finalCiphertexts,
		NumRow:      ctRows,
		NumCol:      ctCols,
		NumBatch:    ctBatch,
	}, ctQKTInvSum, nil
}

// func DeepCopyEvaluator(eval *ckks.Evaluator) *ckks.Evaluator {
// 	params := eval.GetParameters()
// 	evalKeys := rlwe.EvaluationKey{
// 		Rlk:  &(*eval.GetRelinearizationKey()), // 指针拷贝 (数据只读)
// 		Rtks: &(*eval.GetRotationKeySet()),     // 指针拷贝 (数据只读)
// 	}

// 	// 新建Evaluator核心对象（会新建Poly、Buffers、状态）
// 	newEval := ckks.NewEvaluator(*params,)

// 	// Encoder ShallowCopy (独立缓存)
// 	newEval.Encoder = eval.Encoder.ShallowCopy()

// 	// BasisExtender ShallowCopy (独立缓存)
// 	if be := eval.GetBasisExtender(); be != nil {
// 		newEval.SetBasisExtender(be.ShallowCopy())
// 	}

// 	return newEval
// }

type Task struct {
	Start int
	End   int
	Phase int
}

type WorkerPool struct {
	Tasks chan Task
	Wg    sync.WaitGroup
}

func NewWorkerPool(numWorkers int) *WorkerPool {
	pool := &WorkerPool{
		Tasks: make(chan Task, numWorkers*2),
	}
	return pool
}

func (wp *WorkerPool) Run(workerFunc func(Task)) {
	for t := 0; t < cap(wp.Tasks); t++ {
		go func() {
			for task := range wp.Tasks {
				workerFunc(task)
				wp.Wg.Done()
			}
		}()
	}
}

func (wp *WorkerPool) AddTask(task Task) {
	wp.Wg.Add(1)
	wp.Tasks <- task
}

func (wp *WorkerPool) Wait() {
	wp.Wg.Wait()
	close(wp.Tasks)
}

func ComputeAttentionMT1_WorkerPool(
	ctQ, ctK, ctV *he.CiphertextMatrices,
	modelParams *configs.ModelParams,
	ckksParams *ckks.Parameters,
	ecd *ckks.Encoder,
	enc *rlwe.Encryptor,
	eval *ckks.Evaluator,
	btpEval *bootstrapping.Evaluator,
	numThreads int,
) (*he.CiphertextMatrices, *rlwe.Ciphertext, error) {
	// totalTime := time.Now()

	realRow := modelParams.NumRealRow
	ctBatch := ctQ.NumBatch
	ctRows := ctQ.NumRow
	ctCols := ctQ.NumCol

	babyStep := modelParams.BabyStep
	giantStep := modelParams.GiantStep
	baseLen := modelParams.NumBatch

	maskMat := utils.GenerateSpecialSquareMatrix(modelParams)

	babySteps := make([]int, babyStep)
	giantSteps := make([]int, giantStep)
	for i := 0; i < babyStep; i++ {
		babySteps[i] = i * baseLen
	}
	for i := 0; i < giantStep; i++ {
		giantSteps[i] = -i * baseLen * babyStep
	}

	// start := time.Now()

	ctQRotated := matrix.RotateCiphertextMatricesHoistingMT(ctQ, giantSteps, eval, numThreads)
	ctKRotated := matrix.RotateCiphertextMatricesHoistingMT(ctK, babySteps, eval, numThreads)
	ctVRotated := matrix.RotateCiphertextMatricesHoistingMT(ctV, babySteps, eval, numThreads)
	// Precompute constants and rotated matrices...

	workerPool := NewWorkerPool(numThreads)

	var (
		ctQKTCiphertext []*rlwe.Ciphertext
		ctQKTAdd        *rlwe.Ciphertext
		ctVarScale      *rlwe.Ciphertext
		ctExpStore      []*rlwe.Ciphertext
		ctAddAllbyRow   *rlwe.Ciphertext
		ctApproxMaxRot  map[int]*rlwe.Ciphertext
		// mu              sync.Mutex
	)

	// Initialization
	ctQKTCiphertext = make([]*rlwe.Ciphertext, ctRows)
	ctQKTAdd = ckks.NewCiphertext(*ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
	ctVarScale = ckks.NewCiphertext(*ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
	ctExpStore = make([]*rlwe.Ciphertext, ctRows)
	ctAddAllbyRow = ckks.NewCiphertext(*ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
	ctApproxMaxRot = make(map[int]*rlwe.Ciphertext)

	// --- Phase 1 ---
	workerPool.Run(func(task Task) {
		localEval := eval.ShallowCopy()
		for i := task.Start; i < task.End; i++ {
			localCtAddGiantStep := ckks.NewCiphertext(*ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
			for j := 0; j < babyStep; j++ {
				rowIdx := i*babyStep + j
				if rowIdx >= ctRows {
					break
				}
				ct, _ := matrix.CiphertextMatricesMultiplyCiphertextMatricesThenAdd(ctQRotated[i], ctKRotated[j], ckksParams, localEval)
				localEval.Add(localCtAddGiantStep, ct, localCtAddGiantStep)

				ctQKTCiphertext[rowIdx] = ct

			}
			ctAddRotate, _ := localEval.RotateNew(localCtAddGiantStep, i*baseLen*babyStep)

			eval.Add(ctQKTAdd, ctAddRotate, ctQKTAdd)

		}
	})

	chunkSize := (giantStep + numThreads - 1) / numThreads
	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := min((t+1)*chunkSize, giantStep)
		if start >= end {
			continue
		}
		workerPool.AddTask(Task{Start: start, End: end, Phase: 1})
	}
	workerPool.Wait()

	// Elapsed := time.Since(start)
	// fmt.Printf("Q*K^T time :%s\n", Elapsed)

	// start = time.Now()
	ctMean, _ := eval.MulRelinNew(ctQKTAdd, 1/float64(realRow))
	eval.Rescale(ctMean, ctMean)
	ctQKTAddRot, _ := eval.RotateHoistedNew(ctQKTAdd, giantSteps)

	constantValue := 2.08 //2.08
	varScale := math.Sqrt(0.1*(1.0/float64(realRow))*constantValue) * (1.0 / float64(realRow))
	constValue := 1.25 * constantValue

	ctVarScale = ckks.NewCiphertext(*ckksParams, ctMean.Degree(), ctMean.Level())

	// --- Phase 2: Variance Scaling ---
	workerPool = NewWorkerPool(numThreads)
	workerPool.Run(func(task Task) {
		localEval := eval.ShallowCopy()
		for i := task.Start; i < task.End; i++ {
			localCtAddGiantStep := ckks.NewCiphertext(*ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
			for j := 0; j < babyStep; j++ {
				rowIdx := i*babyStep + j
				if rowIdx >= ctRows {
					break
				}
				ctQKTScale, _ := localEval.MulNew(ctQKTCiphertext[rowIdx], realRow)
				ctSub, _ := localEval.SubNew(ctQKTScale, ctQKTAddRot[-i*baseLen*babyStep])
				maskVec := utils.ExtractAndRepeatDiagonal(maskMat, rowIdx, modelParams.NumBatch)
				for idx := range maskVec {
					maskVec[idx] *= varScale
				}
				maskVecRot := utils.RotateSliceNew(maskVec, -i*baseLen*babyStep)
				localEval.MulRelin(ctSub, maskVecRot, ctSub)
				localEval.Rescale(ctSub, ctSub)
				localEval.MulRelin(ctSub, ctSub, ctSub)
				localEval.Rescale(ctSub, ctSub)
				localEval.Add(localCtAddGiantStep, ctSub, localCtAddGiantStep)
			}
			ctAddRotate, _ := localEval.RotateNew(localCtAddGiantStep, i*baseLen*babyStep)

			eval.Add(ctVarScale, ctAddRotate, ctVarScale)

		}
	})
	chunkSize = (giantStep + numThreads - 1) / numThreads
	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := min((t+1)*chunkSize, giantStep)
		if start >= end {
			continue
		}
		workerPool.AddTask(Task{Start: start, End: end, Phase: 2})
	}
	workerPool.Wait()
	ctApproxStd, _ := eval.AddNew(ctVarScale, constValue)
	ctApproxMax, _ := eval.AddNew(ctApproxStd, ctMean)
	ctApproxMaxRot, _ = eval.RotateHoistedNew(ctApproxMax, giantSteps)
	// Elapsed = time.Since(start)
	// fmt.Printf("Approx Max time :%s\n", Elapsed)

	// start = time.Now()
	// --- Phase 3: Compute Exp(QK-Max), Masking, Sum Accumulate ---
	workerPool = NewWorkerPool(numThreads)
	workerPool.Run(func(task Task) {
		localEval := eval.ShallowCopy()
		for i := task.Start; i < task.End; i++ {
			localCtAddGiantStep := ckks.NewCiphertext(*ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
			for j := 0; j < babyStep; j++ {
				rowIdx := i*babyStep + j
				if rowIdx >= ctRows {
					break
				}
				ctSub, _ := localEval.SubNew(ctQKTCiphertext[rowIdx], ctApproxMaxRot[-i*baseLen*babyStep])
				ctExp := softmax.CiphertextExpChebyshev(ctSub, ckksParams, localEval, modelParams.ExpMinValue, modelParams.ExpMaxValue, modelParams.ExpDegree)
				maskVec := utils.ExtractAndRepeatDiagonal(maskMat, rowIdx, modelParams.NumBatch)
				maskVecRot := utils.RotateSliceNew(maskVec, -i*baseLen*babyStep)
				localEval.MulRelin(ctExp, maskVecRot, ctExp)
				localEval.Rescale(ctExp, ctExp)

				ctExpStore[rowIdx] = ctExp

				localEval.Add(localCtAddGiantStep, ctExp, localCtAddGiantStep)
			}
			ctAddRotate, _ := localEval.RotateNew(localCtAddGiantStep, i*baseLen*babyStep)

			eval.Add(ctAddAllbyRow, ctAddRotate, ctAddAllbyRow)

		}
	})
	chunkSize = (giantStep + numThreads - 1) / numThreads
	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := min((t+1)*chunkSize, giantStep)
		if start >= end {
			continue
		}
		workerPool.AddTask(Task{Start: start, End: end, Phase: 3})
	}
	workerPool.Wait()
	// Elapsed = time.Since(start)
	// fmt.Printf("Exp time :%s\n", Elapsed)

	// start = time.Now()
	// --- Phase 4: Bootstrap & Inverse ---
	var ctQKTInvSum *rlwe.Ciphertext
	var btsErr error
	var btsDone sync.WaitGroup
	btsDone.Add(1)
	go func() {
		defer btsDone.Done()
		localEval := eval.ShallowCopy()
		ctSumBts, err := btpEval.Bootstrap(ctAddAllbyRow)
		if err != nil {
			btsErr = err
			return
		}
		ctQKTInvSum = softmax.CiphertextInverse(ctSumBts, ckksParams, localEval, modelParams.InvMinValue, modelParams.InvMaxValue, modelParams.InvDegree, modelParams.InvIter)
	}()

	newCiphertexts := make([]*rlwe.Ciphertext, ctCols)
	for k := 0; k < ctCols; k++ {
		newCiphertexts[k] = ckks.NewCiphertext(*ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
	}

	workerPool = NewWorkerPool(numThreads)
	workerPool.Run(func(task Task) {
		localEval := eval.ShallowCopy()
		for i := task.Start; i < task.End; i++ {
			localQKV := &he.CiphertextMatrices{
				Ciphertexts: make([]*rlwe.Ciphertext, ctCols),
				NumBatch:    ctBatch,
				NumRow:      ctRows,
				NumCol:      ctCols,
			}
			for k := 0; k < ctCols; k++ {
				localQKV.Ciphertexts[k] = ckks.NewCiphertext(*ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
			}
			for j := 0; j < babyStep; j++ {
				rowIdx := i*babyStep + j
				if rowIdx >= ctRows {
					break
				}
				matrix.CiphertextMatricesMultiplyCiphertextAddToRes(ctVRotated[j], ctExpStore[rowIdx], localQKV, ckksParams, localEval)
			}
			QKVRotKi := matrix.RotateCiphertextMatrices(localQKV, i*baseLen*babyStep, localEval)

			for k := 0; k < ctCols; k++ {
				eval.Add(newCiphertexts[k], QKVRotKi.Ciphertexts[k], newCiphertexts[k])
			}

		}
	})

	chunkSize = (giantStep + numThreads - 1) / numThreads
	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := min((t+1)*chunkSize, giantStep)
		if start >= end {
			continue
		}
		workerPool.AddTask(Task{Start: start, End: end, Phase: 4})
	}
	workerPool.Wait()

	btsDone.Wait()
	if btsErr != nil {
		return nil, nil, btsErr
	}
	// Elapsed = time.Since(start)
	// fmt.Printf("BtsInvAndC*V time :%s\n", Elapsed)

	// start = time.Now()
	// --- Phase 5: Multiply by Inverse Sum ---
	workerPool = NewWorkerPool(numThreads)
	workerPool.Run(func(task Task) {
		localEval := eval.ShallowCopy()
		for i := task.Start; i < task.End; i++ {
			ct, _ := localEval.MulRelinNew(newCiphertexts[i], ctQKTInvSum)
			localEval.Rescale(ct, ct)

			newCiphertexts[i] = ct

		}
	})
	chunkSize = (ctCols + numThreads - 1) / numThreads
	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := min((t+1)*chunkSize, ctCols)
		if start >= end {
			continue
		}
		workerPool.AddTask(Task{Start: start, End: end, Phase: 5})
	}
	workerPool.Wait()
	// Elapsed = time.Since(start)
	// fmt.Printf("Reciprocal time :%s\n", Elapsed)

	// totalElapsed := time.Since(totalTime)
	// fmt.Printf("total time :%s\n", totalElapsed)

	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ctBatch,
		NumRow:      ctRows,
		NumCol:      ctCols,
	}, ctQKTInvSum, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
