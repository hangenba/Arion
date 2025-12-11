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

	/* -------- 核心计算: BSGS 乘法 (Phase 2) -------- */
	// 这里使用 Worker Pool 模式，解决 GiantStep < NumThreads 导致的负载不均问题

	// 1. 准备结果容器
	res := make([]*rlwe.Ciphertext, rows)

	// 2. 定义任务结构 (Giant索引, Baby索引, 结果数组的目标索引)
	type workTask struct {
		gIdx      int // index in qRot
		bIdx      int // index in kRot
		targetRow int // index in res
	}

	// 3. 创建通道
	// taskChan 缓冲设为 rows，足以容纳所有任务，避免发送阻塞
	taskChan := make(chan workTask, rows)
	errChan := make(chan error, 1) // 用于捕获错误

	var wg sync.WaitGroup

	// 4. 启动 Workers (消费者)
	// 无论 giantStep 是多少，这里都会启动 numThreads 个协程全力工作
	for w := 0; w < numThreads; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			// [关键] 每个线程拥有独立的 Evaluator，避免内存竞争
			localEval := eval.ShallowCopy()

			for task := range taskChan {
				// 如果已有报错，停止处理后续任务 (可选优化)
				if len(errChan) > 0 {
					continue
				}

				// 执行矩阵乘法累加
				// 注意：这里使用 qRot[task.gIdx] 和 kRot[task.bIdx]
				val, err := matrix.CiphertextMatricesMultiplyCiphertextMatricesThenAdd(
					qRot[task.gIdx],
					kRot[task.bIdx],
					&params, // 注意这里取地址，因为函数定义通常需要指针
					localEval,
				)

				if err != nil {
					// 尝试记录第一个错误
					select {
					case errChan <- err:
					default:
					}
					continue
				}

				// [关键] 无锁写入
				// 因为每个 task.targetRow 是唯一的，所以这里不需要 Mutex
				res[task.targetRow] = val
			}
		}()
	}

	// 5. 分发任务 (生产者)
	// 将双层循环打平成线性任务流
	// Logic: for i in giant, for j in baby -> task
ProducerLoop:
	for i := 0; i < giant; i++ {
		for j := 0; j < baby; j++ {
			row := i*baby + j

			// 边界检查：如果计算出的行号超出了实际行数，停止分发
			if row >= rows {
				break ProducerLoop
			}

			taskChan <- workTask{
				gIdx:      i,
				bIdx:      j,
				targetRow: row,
			}
		}
	}

	// 6. 等待完成
	close(taskChan) // 关闭通道，通知 worker 没有新任务了
	wg.Wait()       // 等待所有 worker 退出

	// 7. 检查错误
	if len(errChan) > 0 {
		return nil, <-errChan
	}

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
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	eval *ckks.Evaluator,
	numThreads int,
) (*he.CiphertextMatrices, error) {
	ctRows := ctV.NumRow
	ctCols := ctV.NumCol
	ctBatch := ctV.NumBatch

	babyStep := modelParams.BabyStep
	giantStep := modelParams.GiantStep
	baseLen := modelParams.NumBatch

	// 1. 基础检查
	if ctQKT.NumBatch != ctV.NumBatch {
		return nil, fmt.Errorf("ctQKT and ctV must have the same Batch")
	}
	if ctQKT.NumCol != ctV.NumRow {
		return nil, fmt.Errorf("ctQKT Cols must equal ctV Row")
	}

	// startTime := time.Now()

	// 2. 预处理：并行生成所有 BabyStep 的旋转矩阵
	// 使用之前的多线程旋转函数，或者在这里原地并行
	babySteps := make([]int, babyStep)
	for i := range babySteps {
		babySteps[i] = i * baseLen
	}
	// 假设你已经有 matrix.RotateCiphertextMatricesMT，如果没有，请保留原有的循环但加上 goroutine
	ctVRotated := matrix.RotateCiphertextMatricesMT(ctV, babySteps, eval, numThreads)

	// fmt.Printf("rot time:%s\n", time.Since(startTime))

	// -----------------------------------------------------------------
	// 核心优化开始：使用中间桶 (Intermediate Buckets) 解决 GiantStep < Threads
	// -----------------------------------------------------------------

	// 准备中间容器：intermediate[giantIndex][colIndex]
	// 对应单线程代码中的 ctQKV，但我们需要 giantStep 个这样的容器
	intermediate := make([][]*rlwe.Ciphertext, giantStep)
	interLocks := make([]sync.Mutex, giantStep) // 每个 giantStep 一把锁

	// 初始化中间容器 (并行初始化，因为 NewCiphertext 也有开销)
	var initWg sync.WaitGroup
	initWg.Add(giantStep)
	for i := 0; i < giantStep; i++ {
		go func(gIdx int) {
			defer initWg.Done()
			intermediate[gIdx] = make([]*rlwe.Ciphertext, ctCols)
			for k := 0; k < ctCols; k++ {
				intermediate[gIdx][k] = ckks.NewCiphertext(ckksParams, ctQKT.Ciphertexts[0].Degree(), ctQKT.Ciphertexts[0].Level())
			}
		}(i)
	}
	initWg.Wait()

	// 定义 Phase 1 的任务
	type calcTask struct {
		gIdx int // giant step index (i)
		bIdx int // baby step index (j)
		row  int // pre-calculated row index
	}

	taskChan := make(chan calcTask, ctRows) // 缓冲足够大
	errChan := make(chan error, 1)

	var workerWg sync.WaitGroup

	// -----------------------------------------------------------------
	// Phase 1: 并行乘法 (Multiplication) & 分桶累加 (Bucket Accumulation)
	// -----------------------------------------------------------------
	// 启动 Worker 池 (充分利用 numThreads)
	for w := 0; w < numThreads; w++ {
		workerWg.Add(1)
		go func() {
			defer workerWg.Done()
			localEval := eval.ShallowCopy() // 线程独享 evaluator

			for task := range taskChan {
				if len(errChan) > 0 {
					continue
				}

				// 1. 创建临时容器存放本次乘法结果 (避免直接操作共享的 intermediate)
				// 这里创建一个临时的 Matrix 结构来适配接口
				tmpResCiphertexts := make([]*rlwe.Ciphertext, ctCols)
				for k := 0; k < ctCols; k++ {
					// 注意：这里必须是新的 Ciphertext，或者是被置零的
					tmpResCiphertexts[k] = ckks.NewCiphertext(ckksParams, ctQKT.Ciphertexts[0].Degree(), ctQKT.Ciphertexts[0].Level())
				}
				tmpCtQKV := &he.CiphertextMatrices{
					Ciphertexts: tmpResCiphertexts,
					NumBatch:    ctBatch,
					NumRow:      ctRows,
					NumCol:      ctCols,
				}

				// 2. 执行计算：tmp = ctVRotated[j] * ctQKT[row]
				// 对应原代码：matrix.CiphertextMatricesMultiplyCiphertextAddToRes(...)
				err := matrix.CiphertextMatricesMultiplyCiphertextAddToRes(
					ctVRotated[task.bIdx],
					ctQKT.Ciphertexts[task.row],
					tmpCtQKV,
					&ckksParams,
					localEval,
				)

				if err != nil {
					select {
					case errChan <- err:
					default:
					}
					continue
				}

				// 3. 累加到对应的 Giant 桶 (Critical Section)
				// 虽然加锁了，但因为计算(上一步)很慢，加锁(这一步)很快，所以冲突极小
				interLocks[task.gIdx].Lock()
				for k := 0; k < ctCols; k++ {
					// intermediate[gIdx] += tmpRes
					localEval.Add(intermediate[task.gIdx][k], tmpResCiphertexts[k], intermediate[task.gIdx][k])
				}
				interLocks[task.gIdx].Unlock()
			}
		}()
	}

	// 分发任务：打平双层循环
	for i := 0; i < giantStep; i++ {
		for j := 0; j < babyStep; j++ {
			row := i*babyStep + j
			if row >= ctRows {
				break
			}
			taskChan <- calcTask{
				gIdx: i,
				bIdx: j,
				row:  row,
			}
		}
	}
	close(taskChan)
	workerWg.Wait() // 等待 Phase 1 完成

	if len(errChan) > 0 {
		return nil, <-errChan
	}

	// -----------------------------------------------------------------
	// Phase 2: 并行旋转 (Rotation) & 最终合并 (Final Merge)
	// -----------------------------------------------------------------
	// 现在 intermediate[i] 已经包含了第 i 个 giant step 的完整 Sum(V * Q)，只差旋转了。

	newCiphertexts := make([]*rlwe.Ciphertext, ctCols)
	// 初始化结果容器
	for k := 0; k < ctCols; k++ {
		newCiphertexts[k] = ckks.NewCiphertext(ckksParams, ctQKT.Ciphertexts[0].Degree(), ctQKT.Ciphertexts[0].Level())
	}

	// 为了最终写入安全，给每列一把锁 (或者使用原子操作，但FHE对象通常用锁)
	finalLocks := make([]sync.Mutex, ctCols)

	var rotWg sync.WaitGroup

	// 简单的并行：遍历每个 giant step
	// 这里的任务数是 giantStep。如果 giantStep < numThreads，这里可能跑不满。
	// 但因为旋转比乘法快得多，且这是收尾阶段，通常可以接受。
	// 如果想极致优化，可以将 (giantStep * ctCols) 打平。下面演示打平版本：

	rotTaskChan := make(chan int, giantStep) // 只传 giant index

	for w := 0; w < numThreads; w++ {
		rotWg.Add(1)
		go func() {
			defer rotWg.Done()
			localEval := eval.ShallowCopy()

			for i := range rotTaskChan {
				// 构造临时的 CiphertextMatrices 以复用 Rotate 接口
				localMat := &he.CiphertextMatrices{
					Ciphertexts: intermediate[i],
					NumBatch:    ctBatch,
					NumRow:      ctRows,
					NumCol:      ctCols,
				}

				// 旋转：Rotate(intermediate[i], i * baseLen * babyStep)
				// 对应原代码：QKVRotKi := matrix.RotateCiphertextMatrices(...)
				rotStep := i * baseLen * babyStep
				QKVRotKi := matrix.RotateCiphertextMatrices(localMat, rotStep, localEval)

				// 累加到最终结果
				for k := 0; k < ctCols; k++ {
					finalLocks[k].Lock()
					localEval.Add(newCiphertexts[k], QKVRotKi.Ciphertexts[k], newCiphertexts[k])
					finalLocks[k].Unlock()
				}
			}
		}()
	}

	// 分发 Phase 2 任务
	for i := 0; i < giantStep; i++ {
		rotTaskChan <- i
	}
	close(rotTaskChan)
	rotWg.Wait()

	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ctBatch,
		NumRow:      ctRows,
		NumCol:      ctCols,
	}, nil
}
