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

func RunThreeMatricMultiplyNormal(modelParams *configs.ModelParams, ckksParams ckks.Parameters, ecd *ckks.Encoder, enc *rlwe.Encryptor, eval *ckks.Evaluator, dec *rlwe.Decryptor) {

	fmt.Println("-------------------- Test Three Matric Multiply Normal --------------------")
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
	res1, err := CiphertextMatrixMultiplyCiphertextMatrixToHalveiShoup(ctA, ctB, ckksParams, eval)
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
	res2, err := CiphertextMatrixHSMultiplyCiphertextMatrix(res1, ctC, ckksParams, eval)
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
	fmt.Printf(" * Total Time: %v\n", totalElapsed)
	fmt.Printf(" 	* CiphertextMatrixMultiplyCiphertextMatrixToHalveiShoup Time: %v\n", elapsed1)
	fmt.Printf(" 	* CiphertextMatrixHSMultiplyCiphertextMatrix Time: %v\n", elapsed2)

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
	fmt.Printf(" * (A*B^T)*C最大误差: %e\n", maxErr2)

	// 打印内存峰值
	fmt.Printf(" * 内存峰值 (HeapAlloc): %.2f MB\n", float64(peakAlloc)/(1024*1024))
	fmt.Printf(" * 内存峰值 (HeapSys): %.2f MB\n", float64(peakSys)/(1024*1024))

}

/*
 * CiphertextMatrixMultiplyCiphertextMatrixToHalveiShoup
 * Input:  PublicParametersKeys,ctMatrix1,ctMatrix2 CiphertextMatrix
 * Output: CiphertextMatrix,error
 * Compute:ct(batch,a,b) X [ctMatrix(batch,b,a)^T --> ctMatrixT(batch,a,b)]--> ctMatrixNew(batch,a,a)
 * 1Mul
 */
func CiphertextMatrixMultiplyCiphertextMatrixToHalveiShoup(ct1, ct2 *he.CiphertextMatrices, ckksParams ckks.Parameters, eval *ckks.Evaluator) (*he.CiphertextMatrices, error) {

	// 返回密文张量1维数
	ct1NumRow := ct1.NumRow
	ct1NumCol := ct1.NumCol
	ct1NumBatch := ct1.NumBatch
	// fmt.Printf("Ciphertext1 Matrix Rows:%d, Cols:%d, Depths:%d\n", ct1Rows, ct1Cols, ct1Depth)

	// 返回密文张量2维数
	ct2NumRow := ct2.NumRow
	ct2NumCol := ct2.NumCol
	ct2NumBatch := ct2.NumBatch
	// fmt.Printf("Ciphertext2 Matrix Rows:%d, Cols:%d, Depths:%d\n", ct2Rows, ct2Cols, ct2Depth)

	// 判断条件
	if ct1NumRow != ct2NumRow || ct1NumCol != ct2NumCol || ct1NumBatch != ct2NumBatch {
		return nil, fmt.Errorf("can not multiply ctMatrix1 and ctMatrix2 transpose to Halevi-Shoup encodeing")
	}

	startTime := time.Now()
	ct2Rot := make([]*he.CiphertextMatrices, ct2NumRow)
	for i := 0; i < ct2NumRow; i++ {
		ctRot := matrix.RotateCiphertextMatrices(ct2, i*ct1NumBatch, eval)
		ct2Rot[i] = ctRot
	}
	elapsed := time.Since(startTime)
	fmt.Printf("rot time:%s\n", elapsed)

	// 进行密文乘法
	newCiphertexts := make([]*rlwe.Ciphertext, ct2NumRow)
	for i := 0; i < ct2NumRow; i++ {
		ctTmp := ckks.NewCiphertext(ckksParams, ct1.Ciphertexts[0].Degree(), ct1.Ciphertexts[0].Level())
		for j := 0; j < ct1NumCol; j++ {
			err := eval.MulRelinThenAdd(ct1.Ciphertexts[j], ct2Rot[i].Ciphertexts[j], ctTmp)
			if err != nil {
				panic(err)
			}
		}
		if err := eval.Rescale(ctTmp, ctTmp); err != nil {
			panic(err)
		}
		newCiphertexts[i] = ctTmp
	}

	// 返回结果
	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ct1NumBatch,
		NumRow:      ct1NumRow,
		NumCol:      ct1NumRow,
	}, nil
}

/*
 * CiphertextMatrixHSMultiplyCiphertextMatrix
 * Input:  PublicParametersKeys,ctMatrix1,ctMatrix2 CiphertextMatrix
 * Output: CiphertextMatrix,error
 * Compute:
 	* ctMatrix1 encoding by Halevi-Shoup
 	* ctMatrix2 encoding by cols
 * 1Mul
*/
func CiphertextMatrixHSMultiplyCiphertextMatrix(ct1, ct2 *he.CiphertextMatrices, ckksParams ckks.Parameters, eval *ckks.Evaluator) (*he.CiphertextMatrices, error) {

	// 返回密文张量1维数
	ct1Row := ct1.NumRow
	ct1Col := ct1.NumCol
	ct1Batch := ct1.NumBatch
	// fmt.Printf("Ciphertext1 Matrix Rows:%d, Cols:%d, Batch:%d\n", ct1Row, ct1Col, ct1Batch)

	// 返回密文张量2维数
	ct2Row := ct2.NumRow
	ct2Col := ct2.NumCol
	ct2Batch := ct2.NumBatch
	// fmt.Printf("Ciphertext2 Matrix Rows:%d, Cols:%d, Batch:%d\n", ct2Row, ct2Col, ct2Batch)

	// 判断条件H-S一定是一个方阵，Matrix1.Depth等于Matrix2.cols
	if ct1Row != ct1Col || ct1Col != ct2Row || ct1Batch != ct2Batch {
		return nil, fmt.Errorf("can not multiply ctMatrix1HS and ctMatrix2 to columns encodeing")
	}

	// 声明ct2Depth条密文
	newCiphertexts := make([]*rlwe.Ciphertext, ct2Col)
	for j := 0; j < ct2Col; j++ {
		newCiphertexts[j] = ckks.NewCiphertext(ckksParams, ct1.Ciphertexts[0].Degree(), ct1.Ciphertexts[0].Level())
	}
	startTime := time.Now()
	ct2Rot := make([]*he.CiphertextMatrices, ct2Row)
	for i := 0; i < ct2Row; i++ {
		ctRot := matrix.RotateCiphertextMatrices(ct2, i*ct1Batch, eval)
		ct2Rot[i] = ctRot
	}
	elapsed := time.Since(startTime)
	fmt.Printf("rot time:%s\n", elapsed)
	// 进行密文乘法
	for i := 0; i < ct2Row; i++ {

		for j := 0; j < ct2Col; j++ {
			// fmt.Printf("stop here %d\n", j)
			// 进行乘法
			// fmt.Println(rotCiperMatrix2.Ciphertexts[j])
			// fmt.Println(ct1.Ciphertexts[j])
			ct, err := eval.MulRelinNew(ct2Rot[i].Ciphertexts[j], ct1.Ciphertexts[i])
			if err != nil {
				panic(err)
			}

			eval.Add(newCiphertexts[j], ct, newCiphertexts[j])
		}
	}

	// 进行rescale
	for j := 0; j < ct2Col; j++ {
		eval.Rescale(newCiphertexts[j], newCiphertexts[j])
	}
	// 返回结果
	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumRow:      ct2Row,
		NumCol:      ct2Col,
		NumBatch:    ct2Batch,
	}, nil
}

func RunThreeMatricMultiplyNormalMT(modelParams *configs.ModelParams, ckksParams ckks.Parameters, ecd *ckks.Encoder, enc *rlwe.Encryptor, eval *ckks.Evaluator, dec *rlwe.Decryptor) {

	threads := 64
	runtime.GOMAXPROCS(threads)
	fmt.Println("-------------------- Test Three Matric Multiply Normal MultiThreads --------------------")
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
	res1, err := CiphertextMatrixMultiplyCiphertextMatrixToHalveiShoupMT(ctA, ctB, ckksParams, eval, threads)
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
	res2, err := CiphertextMatrixHSMultiplyCiphertextMatrixMT(res1, ctC, ckksParams, eval, threads)
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
	fmt.Printf(" * Total Time: %v\n", totalElapsed)
	fmt.Printf(" 	* CiphertextMatrixMultiplyCiphertextMatrixToHalveiShoup Time: %v\n", elapsed1)
	fmt.Printf(" 	* CiphertextMatrixHSMultiplyCiphertextMatrix Time: %v\n", elapsed2)

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
	fmt.Printf(" * (A*B^T)*C最大误差: %e\n", maxErr2)

	// 打印内存峰值
	fmt.Printf(" * 内存峰值 (HeapAlloc): %.2f MB\n", float64(peakAlloc)/(1024*1024))
	fmt.Printf(" * 内存峰值 (HeapSys): %.2f MB\n", float64(peakSys)/(1024*1024))

}

func CiphertextMatrixMultiplyCiphertextMatrixToHalveiShoupMT(
	ct1, ct2 *he.CiphertextMatrices,
	ckksParams ckks.Parameters,
	eval *ckks.Evaluator,
	numThreads int,
) (*he.CiphertextMatrices, error) {

	// 维度检查（直接 panic）
	if ct1.NumRow != ct2.NumRow || ct1.NumCol != ct2.NumCol || ct1.NumBatch != ct2.NumBatch {
		panic("dimension mismatch in HS multiply")
	}
	rows, cols, batch := ct1.NumRow, ct1.NumCol, ct1.NumBatch

	// 预旋转
	steps := make([]int, rows)
	for i := range steps {
		steps[i] = i * batch
	}
	ct2Rot := matrix.RotateCiphertextMatricesMT(ct2, steps, eval, numThreads)

	out := make([]*rlwe.Ciphertext, rows)
	blk := (rows + numThreads - 1) / numThreads

	var wg sync.WaitGroup
	var mu sync.Mutex

	for t := 0; t < numThreads; t++ {
		st, ed := t*blk, min((t+1)*blk, rows)
		if st >= ed {
			continue
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			locEval := eval.ShallowCopy()

			for i := s; i < e; i++ {
				ctTmp := ckks.NewCiphertext(ckksParams,
					ct1.Ciphertexts[0].Degree(), ct1.Ciphertexts[0].Level())

				for j := 0; j < cols; j++ {
					if err := locEval.MulRelinThenAdd(
						ct1.Ciphertexts[j], ct2Rot[i].Ciphertexts[j], ctTmp); err != nil {
						panic(err)
					}
				}
				if err := locEval.Rescale(ctTmp, ctTmp); err != nil {
					panic(err)
				}

				mu.Lock()
				out[i] = ctTmp
				mu.Unlock()
			}
		}(st, ed)
	}
	wg.Wait()

	return &he.CiphertextMatrices{
		Ciphertexts: out,
		NumBatch:    batch,
		NumRow:      rows,
		NumCol:      rows,
	}, nil
}

func CiphertextMatrixHSMultiplyCiphertextMatrixMT(
	ctHS, ctV *he.CiphertextMatrices,
	ckksParams ckks.Parameters,
	eval *ckks.Evaluator,
	numThreads int,
) (*he.CiphertextMatrices, error) {

	if ctHS.NumRow != ctHS.NumCol || ctHS.NumCol != ctV.NumRow || ctHS.NumBatch != ctV.NumBatch {
		panic("dimension mismatch in HS·V multiply")
	}
	rows, cols, batch := ctV.NumRow, ctV.NumCol, ctV.NumBatch

	// 预旋转
	step := make([]int, rows)
	for i := range step {
		step[i] = i * batch
	}
	ctVRot := matrix.RotateCiphertextMatricesMT(ctV, step, eval, numThreads)

	out := make([]*rlwe.Ciphertext, cols)
	for j := range out {
		out[j] = ckks.NewCiphertext(ckksParams,
			ctHS.Ciphertexts[0].Degree(), ctHS.Ciphertexts[0].Level())
	}

	// 行并行 / 列并行二选一
	var wg sync.WaitGroup
	var mu sync.Mutex

	if rows >= numThreads {
		blk := (rows + numThreads - 1) / numThreads
		for t := 0; t < numThreads; t++ {
			st, ed := t*blk, min((t+1)*blk, rows)
			if st >= ed {
				continue
			}

			wg.Add(1)
			go func(s, e int) {
				defer wg.Done()
				locEval := eval.ShallowCopy()
				loc := make([]*rlwe.Ciphertext, cols)
				for j := range loc {
					loc[j] = ckks.NewCiphertext(ckksParams,
						ctHS.Ciphertexts[0].Degree(), ctHS.Ciphertexts[0].Level())
				}

				for i := s; i < e; i++ {
					for j := 0; j < cols; j++ {
						ct, _ := locEval.MulRelinNew(ctVRot[i].Ciphertexts[j], ctHS.Ciphertexts[i])
						locEval.Add(loc[j], ct, loc[j])
					}
				}
				for j := 0; j < cols; j++ {
					_ = locEval.Rescale(loc[j], loc[j])
				}

				mu.Lock()
				for j := 0; j < cols; j++ {
					eval.Add(out[j], loc[j], out[j])
				}
				mu.Unlock()
			}(st, ed)
		}
	} else { // 列并行
		blk := (cols + numThreads - 1) / numThreads
		for t := 0; t < numThreads; t++ {
			st, ed := t*blk, min((t+1)*blk, cols)
			if st >= ed {
				continue
			}

			wg.Add(1)
			go func(s, e int) {
				defer wg.Done()
				locEval := eval.ShallowCopy()
				for j := s; j < e; j++ {
					for i := 0; i < rows; i++ {
						ct, _ := locEval.MulRelinNew(ctVRot[i].Ciphertexts[j], ctHS.Ciphertexts[i])
						locEval.Add(out[j], ct, out[j])
					}
					_ = locEval.Rescale(out[j], out[j])
				}
			}(st, ed)
		}
	}
	wg.Wait()

	return &he.CiphertextMatrices{
		Ciphertexts: out,
		NumRow:      rows,
		NumCol:      cols,
		NumBatch:    batch,
	}, nil
}
