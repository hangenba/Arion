package matmul

import (
	"Arion/configs"
	"Arion/pkg/ecdmat"
	"Arion/pkg/he"
	"Arion/pkg/math/matrix"
	"Arion/pkg/utils"
	"fmt"
	"math/rand/v2"
	"runtime"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"gonum.org/v1/gonum/mat"
)

func RunThreeMatricMultiplyBSGS(modelParams *configs.ModelParams, ckksParams ckks.Parameters, ecd *ckks.Encoder, enc *rlwe.Encryptor, eval *ckks.Evaluator, dec *rlwe.Decryptor) {

	fmt.Println("-------------------- Test Three Matric Multiply BSGS --------------------")
	fmt.Printf(" * Setting: LogN:%d , Batch:%d , Row:%d , Col:%d\n", ckksParams.LogN(), modelParams.NumBatch, modelParams.NumRow, modelParams.NumCol)
	numRow := modelParams.NumRow
	numCol := modelParams.NumCol

	// 生成三个明文矩阵
	valuesA := make([]float64, numRow*numCol)
	valuesB := make([]float64, numRow*numCol)
	valuesC := make([]float64, numRow*numCol)
	for i := range valuesA {
		valuesA[i] = rand.Float64()*2 - 1
		valuesB[i] = rand.Float64()*2 - 1
		valuesC[i] = rand.Float64()*2 - 1
	}
	matA := mat.NewDense(numRow, numCol, valuesA)
	matB := mat.NewDense(numRow, numCol, valuesB)
	matC := mat.NewDense(numRow, numCol, valuesC)

	// 编码和加密
	batchMatsA := utils.MatrixToBatchMats(matA, modelParams)
	batchMatsB := utils.MatrixToBatchMats(matB, modelParams)
	batchMatsC := utils.MatrixToBatchMats(matC, modelParams)
	encodedA := ecdmat.EncodeDense(batchMatsA, modelParams)
	encodedB := ecdmat.EncodeDense(batchMatsB, modelParams)
	encodedC := ecdmat.EncodeDense(batchMatsC, modelParams)

	ctA, err := he.EncryptInputMatrices(encodedA, modelParams, ckksParams, enc, ecd)
	if err != nil {
		panic(fmt.Sprintf("加密A失败: %v", err))
	}
	ctB, err := he.EncryptInputMatrices(encodedB, modelParams, ckksParams, enc, ecd)
	if err != nil {
		panic(fmt.Sprintf("加密B失败: %v", err))
	}
	ctC, err := he.EncryptInputMatrices(encodedC, modelParams, ckksParams, enc, ecd)
	if err != nil {
		panic(fmt.Sprintf("加密C失败: %v", err))
	}

	// 获取初始内存状态
	var memStart, memCurrent runtime.MemStats
	runtime.ReadMemStats(&memStart)
	peakAlloc := memStart.Alloc // 初始堆内存
	peakSys := memStart.Sys     // 初始系统内存分配

	// 密文计算总时间计时开始
	totalStart := time.Now()

	// 第一个函数计时
	start1 := time.Now()
	res1, err := CiphertextMatrixMultiplyCiphertextMatrixToHalveiShoupBSGS(ctA, ctB, modelParams, ckksParams, eval)
	if err != nil {
		panic(fmt.Sprintf("CiphertextMatrixMultiplyCiphertextMatrixToHalveiShoup失败: %v", err))
	}
	elapsed1 := time.Since(start1)

	// 更新内存状态
	runtime.ReadMemStats(&memCurrent)
	if memCurrent.Alloc > peakAlloc {
		peakAlloc = memCurrent.Alloc
	}
	if memCurrent.Sys > peakSys {
		peakSys = memCurrent.Sys
	}

	// 第二个函数计时
	start2 := time.Now()
	res2, err := CiphertextMatrixHSMultiplyCiphertextMatrixBSGS(res1, ctC, modelParams, ckksParams, eval)
	if err != nil {
		panic(fmt.Sprintf("CiphertextMatrixHSMultiplyCiphertextMatrix失败: %v", err))
	}
	elapsed2 := time.Since(start2)

	// 再次更新内存状态
	runtime.ReadMemStats(&memCurrent)
	if memCurrent.Alloc > peakAlloc {
		peakAlloc = memCurrent.Alloc
	}
	if memCurrent.Sys > peakSys {
		peakSys = memCurrent.Sys
	}

	// 总密文计算时间
	totalElapsed := time.Since(totalStart)
	fmt.Printf("* Total Time: %v\n", totalElapsed)
	fmt.Printf("	* CiphertextMatrixMultiplyCiphertextMatrixToHalveiShoup Time: %v\n", elapsed1)
	fmt.Printf("	* CiphertextMatrixHSMultiplyCiphertextMatrix Time: %v\n", elapsed2)

	// 解密
	decRes2, err := he.DecryptCiphertextMatrices(res2, modelParams, ckksParams, dec, ecd)
	if err != nil {
		panic(fmt.Sprintf("解密res2失败: %v", err))
	}
	ptRes2 := ecdmat.DecodeDense(decRes2, modelParams)
	// fmt.Println("res2:")
	// utils.PrintMat(ptRes2[0])

	// 明文计算
	// 1. A * B^T
	matBT := mat.NewDense(numCol, numRow, nil)
	matBT.CloneFrom(matB.T())
	matRes1 := mat.NewDense(numRow, numRow, nil)
	matRes1.Mul(matA, matBT)

	// 2. (A * B^T) * C
	matRes2 := mat.NewDense(numRow, numCol, nil)
	matRes2.Mul(matRes1, matC)

	maxErr2 := 0.0
	for i := 0; i < numRow; i++ {
		for j := 0; j < numCol; j++ {
			diff := ptRes2[0].At(i, j) - matRes2.At(i, j)
			if diff < 0 {
				diff = -diff
			}
			if diff > maxErr2 {
				maxErr2 = diff
			}
		}
	}
	fmt.Printf("* (A*B^T)*C最大误差: %e\n", maxErr2)

	// 打印内存峰值
	fmt.Printf("* 内存峰值 (HeapAlloc): %.2f MB\n", float64(peakAlloc)/(1024*1024))
	fmt.Printf("* 内存峰值 (HeapSys): %.2f MB\n", float64(peakSys)/(1024*1024))

}

func CiphertextMatrixMultiplyCiphertextMatrixToHalveiShoupBSGS(
	ctQ, ctK *he.CiphertextMatrices,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	eval *ckks.Evaluator,
) (*he.CiphertextMatrices, error) {
	ctRows := ctQ.NumRow
	// ctCols := ctQ.NumCol
	ctBatch := ctQ.NumBatch

	babyStep := modelParams.BabyStep
	giantStep := modelParams.GiantStep
	baseLen := modelParams.NumBatch

	// 检查QKV是否一致
	if ctQ.NumBatch != ctK.NumBatch {
		return nil, fmt.Errorf("ctQ, ctK must have the same Batch: ctQ:%d , ctK:%d ", ctQ.NumBatch, ctK.NumBatch)
	}
	if ctQ.NumRow != ctK.NumRow {
		return nil, fmt.Errorf("ctQ, ctK must have the same Row: ctQ:%d , ctK:%d ", ctQ.NumRow, ctK.NumRow)
	}
	if ctQ.NumCol != ctK.NumCol {
		return nil, fmt.Errorf("ctQ, ctK must have the same Col: ctQ:%d , ctK:%d ", ctQ.NumCol, ctK.NumCol)
	}
	startTime := time.Now()
	// fmt.Println(babyStep, giantStep)
	// 生成所有步长
	ctKRotated := make([]*he.CiphertextMatrices, babyStep)
	// babySteps := make([]int, 0, babyStep)
	for i := 0; i < babyStep; i++ {
		// babySteps = append(babySteps, i*baseLen)
		ctRot := matrix.RotateCiphertextMatrices(ctK, i*baseLen, eval)
		ctKRotated[i] = ctRot
	}
	ctQRotated := make([]*he.CiphertextMatrices, giantStep)
	// giantSteps := make([]int, 0, giantStep)
	for i := 0; i < giantStep; i++ {
		// giantSteps = append(giantSteps, -i*baseLen*babyStep)
		ctRot := matrix.RotateCiphertextMatrices(ctQ, -i*baseLen*babyStep, eval)
		ctQRotated[i] = ctRot
	}

	// 使用 hoisting 技术批量旋转所有的密文矩阵

	elapsed := time.Since(startTime)
	fmt.Printf("rot time:%s\n", elapsed)

	ctQKTCiphertext := make([]*rlwe.Ciphertext, ctRows)
	// Step1. Compute Add All Row
	for i := 0; i < giantStep; i++ {
		// 用于存储每个giantStep的结果
		for j := 0; j < babyStep; j++ {
			// 检查是否超出行数
			if i*babyStep+j >= ctRows {
				break // 超出行数，退出循环
			}

			// step1.1: 计算 ctQRotated[i] 和 ctKRotated[j] 的乘积
			ct, err := matrix.CiphertextMatricesMultiplyCiphertextMatricesThenAdd(ctQRotated[i], ctKRotated[j], &ckksParams, eval)
			if err != nil {
				return nil, fmt.Errorf("failed to multiply ctQRotated[%d] and ctKRotated[%d]: %w", i, j, err)
			}
			ctQKTCiphertext[i*babyStep+j] = ct
		}
	}

	return &he.CiphertextMatrices{
		Ciphertexts: ctQKTCiphertext,
		NumBatch:    ctBatch,
		NumRow:      ctRows,
		NumCol:      ctRows,
	}, nil
}

func CiphertextMatrixHSMultiplyCiphertextMatrixBSGS(
	ctQKT, ctV *he.CiphertextMatrices,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	eval *ckks.Evaluator,
) (*he.CiphertextMatrices, error) {
	ctRows := ctV.NumRow
	ctCols := ctV.NumCol
	ctBatch := ctV.NumBatch

	babyStep := modelParams.BabyStep
	giantStep := modelParams.GiantStep
	baseLen := modelParams.NumBatch

	// 检查QKV是否一致
	if ctQKT.NumBatch != ctV.NumBatch {
		return nil, fmt.Errorf("ctQKT and ctV must have the same Batch: ctQKT:%d , ctV: %d", ctQKT.NumBatch, ctV.NumBatch)
	}
	if ctQKT.NumCol != ctV.NumRow {
		return nil, fmt.Errorf("ctQKT Cols must equal ctV Row : ctQKT:%d  ctV: %d", ctQKT.NumCol, ctV.NumRow)
	}

	// fmt.Println(babyStep, giantStep)
	// 生成所有步长

	// 使用 hoisting 技术批量旋转所有的密文矩阵
	startTime := time.Now()
	ctVRotated := make([]*he.CiphertextMatrices, babyStep)
	// babySteps := make([]int, 0, babyStep)
	for i := 0; i < babyStep; i++ {
		// babySteps = append(babySteps, i*baseLen)
		ctRot := matrix.RotateCiphertextMatrices(ctV, i*baseLen, eval)
		ctVRotated[i] = ctRot
	}

	elapsed := time.Since(startTime)
	fmt.Printf("rot time:%s\n", elapsed)

	newCiphertexts := make([]*rlwe.Ciphertext, ctCols)
	for k := 0; k < ctCols; k++ {
		newCiphertexts[k] = ckks.NewCiphertext(ckksParams, ctQKT.Ciphertexts[0].Degree(), ctQKT.Ciphertexts[0].Level())
	}

	for i := 0; i < giantStep; i++ {

		// 这里的ctQKV是一个新的CiphertextMatrices，用于存储 QKV 的结果
		localNewCiphertexts := make([]*rlwe.Ciphertext, ctV.NumCol)
		for k := 0; k < ctV.NumCol; k++ {
			localNewCiphertexts[k] = ckks.NewCiphertext(ckksParams, ctQKT.Ciphertexts[0].Degree(), ctQKT.Ciphertexts[0].Level())
		}
		ctQKV := &he.CiphertextMatrices{
			Ciphertexts: localNewCiphertexts,
			NumBatch:    ctBatch,
			NumRow:      ctRows,
			NumCol:      ctCols,
		}
		for j := 0; j < babyStep; j++ {
			// 检查是否超出行数
			if i*babyStep+j >= ctRows {
				break // 超出行数，退出循环
			}

			// step4: 将Exp(ct - ctApproxMax) * ctVRotated[j] 存入ctQKV
			err := matrix.CiphertextMatricesMultiplyCiphertextAddToRes(ctVRotated[j], ctQKT.Ciphertexts[i*babyStep+j], ctQKV, &ckksParams, eval)
			if err != nil {
				panic(err)

			}

		}
		// 旋转QKV的结果
		QKVRotKi := matrix.RotateCiphertextMatrices(ctQKV, i*baseLen*babyStep, eval)
		// 将QKVRotKi的结果累加到newCiphertexts中
		for k := 0; k < ctCols; k++ {
			err := eval.Add(newCiphertexts[k], QKVRotKi.Ciphertexts[k], newCiphertexts[k])
			if err != nil {
				panic(err)
			}
		}

	}

	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ctBatch,
		NumRow:      ctRows,
		NumCol:      ctCols,
	}, nil
}

// -----------------------------------------------------------------------------
// RunThreeMatricMultiplyBSGS_MT
// -----------------------------------------------------------------------------
func RunThreeMatricMultiplyBSGSMT(
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	ecd *ckks.Encoder,
	enc *rlwe.Encryptor,
	eval *ckks.Evaluator,
	dec *rlwe.Decryptor,
) {

	threads := 64
	runtime.GOMAXPROCS(threads)
	fmt.Println("-------------------- Test Three Matric Multiply BSGS MultiThread --------------------")
	fmt.Printf(" * Setting: LogN:%d , Batch:%d , Row:%d , Col:%d\n", ckksParams.LogN(), modelParams.NumBatch, modelParams.NumRow, modelParams.NumCol)
	numRow := modelParams.NumRow
	numCol := modelParams.NumCol

	// 生成三个明文矩阵
	valuesA := make([]float64, numRow*numCol)
	valuesB := make([]float64, numRow*numCol)
	valuesC := make([]float64, numRow*numCol)
	for i := range valuesA {
		valuesA[i] = rand.Float64()*2 - 1
		valuesB[i] = rand.Float64()*2 - 1
		valuesC[i] = rand.Float64()*2 - 1
	}
	matA := mat.NewDense(numRow, numCol, valuesA)
	matB := mat.NewDense(numRow, numCol, valuesB)
	matC := mat.NewDense(numRow, numCol, valuesC)

	// 编码和加密
	batchMatsA := utils.MatrixToBatchMats(matA, modelParams)
	batchMatsB := utils.MatrixToBatchMats(matB, modelParams)
	batchMatsC := utils.MatrixToBatchMats(matC, modelParams)
	encodedA := ecdmat.EncodeDense(batchMatsA, modelParams)
	encodedB := ecdmat.EncodeDense(batchMatsB, modelParams)
	encodedC := ecdmat.EncodeDense(batchMatsC, modelParams)

	ctA, err := he.EncryptInputMatrices(encodedA, modelParams, ckksParams, enc, ecd)
	if err != nil {
		panic(fmt.Sprintf("加密A失败: %v", err))
	}
	ctB, err := he.EncryptInputMatrices(encodedB, modelParams, ckksParams, enc, ecd)
	if err != nil {
		panic(fmt.Sprintf("加密B失败: %v", err))
	}
	ctC, err := he.EncryptInputMatrices(encodedC, modelParams, ckksParams, enc, ecd)
	if err != nil {
		panic(fmt.Sprintf("加密C失败: %v", err))
	}

	// 获取初始内存状态
	var memStart, memCurrent runtime.MemStats
	runtime.ReadMemStats(&memStart)
	peakAlloc := memStart.Alloc // 初始堆内存
	peakSys := memStart.Sys     // 初始系统内存分配

	// 密文计算总时间计时开始
	totalStart := time.Now()

	// 第一个函数计时
	start1 := time.Now()
	res1, err := CiphertextMatrixMultiplyCiphertextMatrixToHalveiShoupBSGSMT(ctA, ctB, modelParams, ckksParams, eval, threads)
	if err != nil {
		panic(fmt.Sprintf("CiphertextMatrixMultiplyCiphertextMatrixToHalveiShoup失败: %v", err))
	}
	elapsed1 := time.Since(start1)

	// 更新内存状态
	runtime.ReadMemStats(&memCurrent)
	if memCurrent.Alloc > peakAlloc {
		peakAlloc = memCurrent.Alloc
	}
	if memCurrent.Sys > peakSys {
		peakSys = memCurrent.Sys
	}

	// 第二个函数计时
	start2 := time.Now()
	res2, err := CiphertextMatrixHSMultiplyCiphertextMatrixBSGSMT(res1, ctC, modelParams, ckksParams, eval, threads)
	if err != nil {
		panic(fmt.Sprintf("CiphertextMatrixHSMultiplyCiphertextMatrix失败: %v", err))
	}
	elapsed2 := time.Since(start2)

	// 再次更新内存状态
	runtime.ReadMemStats(&memCurrent)
	if memCurrent.Alloc > peakAlloc {
		peakAlloc = memCurrent.Alloc
	}
	if memCurrent.Sys > peakSys {
		peakSys = memCurrent.Sys
	}

	// 总密文计算时间
	totalElapsed := time.Since(totalStart)
	fmt.Printf("* Total Time: %v\n", totalElapsed)
	fmt.Printf("	* CiphertextMatrixMultiplyCiphertextMatrixToHalveiShoupMT Time: %v\n", elapsed1)
	fmt.Printf("	* CiphertextMatrixHSMultiplyCiphertextMatrixMT Time: %v\n", elapsed2)

	// 解密
	decRes2, err := he.DecryptCiphertextMatrices(res2, modelParams, ckksParams, dec, ecd)
	if err != nil {
		panic(fmt.Sprintf("解密res2失败: %v", err))
	}
	ptRes2 := ecdmat.DecodeDense(decRes2, modelParams)
	// fmt.Println("res2:")
	// utils.PrintMat(ptRes2[0])

	// 明文计算
	// 1. A * B^T
	matBT := mat.NewDense(numCol, numRow, nil)
	matBT.CloneFrom(matB.T())
	matRes1 := mat.NewDense(numRow, numRow, nil)
	matRes1.Mul(matA, matBT)

	// 2. (A * B^T) * C
	matRes2 := mat.NewDense(numRow, numCol, nil)
	matRes2.Mul(matRes1, matC)

	maxErr2 := 0.0
	for i := 0; i < numRow; i++ {
		for j := 0; j < numCol; j++ {
			diff := ptRes2[0].At(i, j) - matRes2.At(i, j)
			if diff < 0 {
				diff = -diff
			}
			if diff > maxErr2 {
				maxErr2 = diff
			}
		}
	}
	fmt.Printf("* (A*B^T)*C最大误差: %e\n", maxErr2)

	// 打印内存峰值
	fmt.Printf("* 内存峰值 (HeapAlloc): %.2f MB\n", float64(peakAlloc)/(1024*1024))
	fmt.Printf("* 内存峰值 (HeapSys): %.2f MB\n", float64(peakSys)/(1024*1024))
}

// Q · Kᵀ  ——  二级并行（giantStep × babyStep）
func CiphertextMatrixMultiplyCiphertextMatrixToHalveiShoupBSGSMT(
	ctQ, ctK *he.CiphertextMatrices,
	mp *configs.ModelParams,
	params ckks.Parameters,
	eval *ckks.Evaluator,
	numThreads int,
) (*he.CiphertextMatrices, error) {

	/* -------- 维度检查 -------- */

	rows, baby, giant, base := ctQ.NumRow, mp.BabyStep, mp.GiantStep, mp.NumBatch

	/* -------- 步长 / 旋转 -------- */
	babySteps := make([]int, baby)
	for i := range babySteps {
		babySteps[i] = i * base
	}

	giantSteps := make([]int, giant)
	for i := range giantSteps {
		giantSteps[i] = -i * base * baby
	}

	qRot := matrix.RotateCiphertextMatricesMT(ctQ, giantSteps, eval, numThreads)
	kRot := matrix.RotateCiphertextMatricesMT(ctK, babySteps, eval, numThreads)

	res := make([]*rlwe.Ciphertext, rows)

	/* -------- 线程层级 -------- */
	T1 := giant // 外层 ≤ giantStep
	if T1 > numThreads {
		T1 = numThreads
	}
	if T1 == 0 {
		T1 = 1
	}
	T2 := numThreads / T1 // 内层
	if T2 == 0 {
		T2 = 1
	}

	var wg sync.WaitGroup
	var mu sync.Mutex

	for g := 0; g < T1; g++ {
		wg.Add(1)
		go func(gid int) {
			defer wg.Done()
			lEval := eval.ShallowCopy()
			for gi := gid; gi < giant; gi += T1 {

				// -------- babyStep 级别 --------
				var gWg sync.WaitGroup
				chunk := (baby + T2 - 1) / T2

				for t := 0; t < T2; t++ {
					l := t * chunk
					r := (t + 1) * chunk
					if l >= r || l >= baby {
						continue
					}
					if r > baby {
						r = baby
					}

					gWg.Add(1)
					go func(l, r, gi int) {
						defer gWg.Done()
						le := lEval.ShallowCopy()
						localQRot := qRot[gi].CopyNew()
						for bj := l; bj < r; bj++ {
							localKRot := kRot[bj].CopyNew()
							row := gi*baby + bj
							if row >= rows {
								break
							}

							ct, _ := matrix.CiphertextMatricesMultiplyCiphertextMatricesThenAdd(
								localQRot, localKRot, &params, le)

							mu.Lock()
							res[row] = ct
							mu.Unlock()
						}
					}(l, r, gi)
				}
				gWg.Wait()
			}
		}(g)
	}
	wg.Wait()

	return &he.CiphertextMatrices{
		Ciphertexts: res,
		NumBatch:    ctQ.NumBatch,
		NumRow:      rows,
		NumCol:      rows,
	}, nil
}

// (QKᵀ) · V   ——  二级并行  (giantStep × babyStep)
func CiphertextMatrixHSMultiplyCiphertextMatrixBSGSMT(
	ctQKT, ctV *he.CiphertextMatrices,
	mp *configs.ModelParams,
	params ckks.Parameters,
	eval *ckks.Evaluator,
	numThreads int,
) (*he.CiphertextMatrices, error) {

	rows, cols := ctV.NumRow, ctV.NumCol
	baby, giant, base := mp.BabyStep, mp.GiantStep, mp.NumBatch

	/* 基本检查 */
	if ctQKT.NumBatch != ctV.NumBatch || ctQKT.NumCol != ctV.NumRow {
		return nil, fmt.Errorf("dimension mismatch QKT(%d,%d) vs V(%d,%d)",
			ctQKT.NumBatch, ctQKT.NumCol, ctV.NumRow, ctV.NumCol)
	}

	/* 预旋转 V */
	babySteps := make([]int, baby)
	for i := range babySteps {
		babySteps[i] = i * base
	}
	vRot := matrix.RotateCiphertextMatricesMT(ctV, babySteps, eval, numThreads)

	out := make([]*rlwe.Ciphertext, cols)
	for k := range out {
		out[k] = ckks.NewCiphertext(params, ctQKT.Ciphertexts[0].Degree(), ctQKT.Ciphertexts[0].Level())
	}

	/* 线程层级 */
	T1 := giant
	if T1 > numThreads {
		T1 = numThreads
	}
	if T1 == 0 {
		T1 = 1
	}
	T2 := numThreads / T1
	if T2 == 0 {
		T2 = 1
	}

	var wg sync.WaitGroup
	var mu sync.Mutex

	for g := 0; g < T1; g++ {
		wg.Add(1)
		go func(gid int) {
			defer wg.Done()
			lEval := eval.ShallowCopy()

			// thread-local 累加器
			localSum := func() []*rlwe.Ciphertext {
				cs := make([]*rlwe.Ciphertext, cols)
				for k := range cs {
					cs[k] = ckks.NewCiphertext(params,
						ctQKT.Ciphertexts[0].Degree(), ctQKT.Ciphertexts[0].Level())
				}
				return cs
			}

			for gi := gid; gi < giant; gi += T1 {
				tmp := localSum()

				/* 内层 baby 并行 */
				var gWg sync.WaitGroup
				chunk := (baby + T2 - 1) / T2

				for t := 0; t < T2; t++ {
					l := t * chunk
					r := (t + 1) * chunk
					if l >= r || l >= baby {
						continue
					}
					if r > baby {
						r = baby
					}

					gWg.Add(1)
					go func(l, r, gi int, acc []*rlwe.Ciphertext) {
						defer gWg.Done()
						le := lEval.ShallowCopy()

						localMat := &he.CiphertextMatrices{
							Ciphertexts: make([]*rlwe.Ciphertext, cols),
							NumBatch:    ctV.NumBatch,
							NumRow:      rows,
							NumCol:      cols,
						}
						for k := 0; k < cols; k++ {
							localMat.Ciphertexts[k] = ckks.NewCiphertext(params,
								ctQKT.Ciphertexts[0].Degree(), ctQKT.Ciphertexts[0].Level())
						}

						for bj := l; bj < r; bj++ {
							localVrot := vRot[bj].CopyNew()
							row := gi*baby + bj
							if row >= rows {
								break
							}

							_ = matrix.CiphertextMatricesMultiplyCiphertextAddToRes(
								localVrot, ctQKT.Ciphertexts[row], localMat, &params, le)
						}

						rot := matrix.RotateCiphertextMatrices(localMat, gi*base*baby, le)
						for k := 0; k < cols; k++ {
							le.Add(acc[k], rot.Ciphertexts[k], acc[k])
						}

					}(l, r, gi, tmp)
				}
				gWg.Wait()

				// merge giant-level partial sum
				mu.Lock()
				for k := 0; k < cols; k++ {
					eval.Add(out[k], tmp[k], out[k])
				}
				mu.Unlock()
			}
		}(g)
	}
	wg.Wait()

	return &he.CiphertextMatrices{
		Ciphertexts: out,
		NumBatch:    ctV.NumBatch,
		NumRow:      rows,
		NumCol:      cols,
	}, nil
}
