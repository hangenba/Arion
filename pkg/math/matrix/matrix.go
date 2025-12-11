package matrix

import (
	"Arion/configs"
	"Arion/pkg/he"
	"Arion/pkg/math/activation/softmax"
	"Arion/pkg/utils"
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"gonum.org/v1/gonum/mat"
)

/*
 * CiphertextMatricesMultiplyPlaintextMatrix
 * 输入:  ctMatrices *he.CiphertextMatrices, ptMat mat.Matrix
 * 输出: *he.CiphertextMatrices, error
 * 计算: ctMatrices(a, b, c) × ptMat(c, d) --> ctMatricesNew(a, b, d)
 * 说明: 对每个输出列d，遍历明文矩阵ptMat的每一行c，将ptMat(c, d)扩展为明文向量，与ctMatrices的第c个密文做乘法并累加，得到新的密文列。
 *      适用于批处理密文矩阵与明文权重矩阵的乘法，输出的密文矩阵列数等于明文权重矩阵的列数。
 */

func CiphertextMatricesMultiplyPlaintextMatrix(ctMatrices *he.CiphertextMatrices, modelParams *configs.ModelParams, ptMat mat.Matrix, ckksParams ckks.Parameters, eval *ckks.Evaluator) (*he.CiphertextMatrices, error) {

	// 返回维数
	ctRows := ctMatrices.NumRow
	ctCols := ctMatrices.NumCol
	// ctBatch := ctMatrices.NumBatch
	// ctRealRows := modelParams.NumRealRow
	// fmt.Println("realRow: ", ctRealRows)
	// fmt.Printf("Ciphertext Matrices Batch:%d, Rows:%d, Cols:%d\n", ctBatch, ctRows, ctCols)

	// 确定进行密文矩阵×明文矩阵的维数
	ptRows, ptCols := ptMat.Dims()
	fmt.Printf("Plaintext Matrix Rows:%d, Cols:%d\n", ptRows, ptCols)

	// 实际上，密文的depths必须等于明文的rows，才能继续进行运算；而明文的cols则是运算之后的depths
	if ctCols != ptRows {
		return nil, fmt.Errorf("the ciphertext Matrices cannot multiply plaintext matrix: expected depth %d, got %d", ptRows, ptCols)
	}

	// 进行计算
	newCiphertexts := make([]*rlwe.Ciphertext, ptCols)
	for i := 0; i < ptCols; i++ {
		ct := ckks.NewCiphertext(ckksParams, ctMatrices.Ciphertexts[0].Degree(), ctMatrices.Ciphertexts[0].Level())
		for j := 0; j < ptRows; j++ {
			// 创建一个明文向量，将ptMat第j行第i列的元素重复ctBatch次
			// ptMulNumSlice := make([]float64, ctBatch*ctRealRows)
			// for k := range ptMulNumSlice {
			// 	ptMulNumSlice[k] = ptMat.At(j, i)
			// }
			ctMatrices.Ciphertexts[j].Scale = ckksParams.DefaultScale()
			// ct.Scale = ckksParams.DefaultScale()
			err := eval.MulThenAdd(ctMatrices.Ciphertexts[j], ptMat.At(j, i), ct)
			if err != nil {
				panic(err)
			}
		}
		if err := eval.Rescale(ct, ct); err != nil {
			panic(err)
		}

		// ct.Scale = ckksParams.DefaultScale()
		newCiphertexts[i] = ct
	}

	// 返回结果
	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ctMatrices.NumBatch,
		NumRow:      ctRows,
		NumCol:      ptCols,
	}, nil
}

/*
 * CiphertextMatricesAddPlaintextMatrix
 * 输入:  ctMatrices *he.CiphertextMatrices, ptMat mat.Matrix
 * 输出: *he.CiphertextMatrices, error
 * 计算: ctMatrices(a, b, c) + ptMat(a, c) --> ctMatricesNew(a, b, c)
 * 说明: 对每个密文矩阵的每个元素，加上明文矩阵对应元素，输出新的密文矩阵。
 *      要求明文矩阵ptMat的行数等于密文矩阵的行数，列数等于密文矩阵的列数。
 */
func CiphertextMatricesAddPlaintextMatrix(ctMatrices *he.CiphertextMatrices, ptMat mat.Matrix, ckksParams ckks.Parameters, eval *ckks.Evaluator) (*he.CiphertextMatrices, error) {
	ctRows := ctMatrices.NumRow
	ctCols := ctMatrices.NumCol
	ctBatch := ctMatrices.NumBatch

	ptRows, ptCols := ptMat.Dims()

	// 检查维度
	if ctCols != ptRows || len(ctMatrices.Ciphertexts) != ctCols {
		return nil, fmt.Errorf("the ciphertext matrices cannot add plaintext matrix: expected (%d,%d), got (%d,%d)", ptRows, ptCols, ctCols, ptCols)
	}

	newCiphertexts := make([]*rlwe.Ciphertext, ctCols)
	for i := 0; i < ctCols; i++ {
		// 构造明文向量，将ptMat第j行第i列的元素重复ctBatch次
		plainVector := make([]float64, ctBatch*ctRows)
		for j := 0; j < ctRows; j++ {
			val := ptMat.At(j, i)
			for b := 0; b < ctBatch; b++ {
				plainVector[j*ctBatch+b] = val
			}
		}
		// 密文加明文
		ct, err := eval.AddNew(ctMatrices.Ciphertexts[i], plainVector)
		if err != nil {
			return nil, err
		}
		newCiphertexts[i] = ct
	}

	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ctBatch,
		NumRow:      ctRows,
		NumCol:      ctCols,
	}, nil
}

/*
 * CiphertextMatricesAddPlainValue
 * 对 CiphertextMatrices 中所有密文加上一个常数 value，返回新的 CiphertextMatrices。
 * 输入：
 *   ctMats - 输入的密文矩阵指针
 *   value  - 需要加到每个密文上的常数（float64）
 *   eval   - CKKS Evaluator
 * 输出：
 *    新的密文矩阵，每个密文都加上了 value
 */
func CiphertextMatricesAddPlainValue(
	ctMats *he.CiphertextMatrices,
	value float64,
	eval *ckks.Evaluator,
) *he.CiphertextMatrices {
	newCiphertexts := make([]*rlwe.Ciphertext, len(ctMats.Ciphertexts))
	for i, ct := range ctMats.Ciphertexts {
		res, err := eval.AddNew(ct, value)
		if err != nil {
			panic(err)
		}
		newCiphertexts[i] = res
	}
	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ctMats.NumBatch,
		NumRow:      ctMats.NumRow,
		NumCol:      ctMats.NumCol,
	}
}

/*
 * CiphertextMatricesMultiplyWeightAndAddBias
 * 输入:  ctMatrices *he.CiphertextMatrices, ptWeight mat.Matrix, ptBias mat.Matrix
 * 输出: *he.CiphertextMatrices, error
 * 计算: ctMatrices(a, b, c) × ptWeight(c, d) + ptBias(d, 1) --> ctMatricesNew(a, b, d)
 * 步骤:
 *   1. 密文矩阵 × 明文权重矩阵
 *   2. + 明文偏置向量（ptBias为[d,1]的mat.Matrix）
 */
func CiphertextMatricesMultiplyWeightAndAddBias(
	ctMatrices *he.CiphertextMatrices,
	ptWeight mat.Matrix,
	ptBias mat.Matrix,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	eval *ckks.Evaluator,
) (*he.CiphertextMatrices, error) {
	// Step 1. 密文矩阵 × 明文权重矩阵
	ctMul, err := CiphertextMatricesMultiplyPlaintextMatrix(ctMatrices, modelParams, ptWeight, ckksParams, eval)
	if err != nil {
		return nil, err
	}

	// Step 2. + 明文偏置向量
	_, biasCols := ptBias.Dims()
	if biasCols != 1 {
		return nil, fmt.Errorf("ptBias 必须为 shape (d, 1)，实际为 (d, %d)", biasCols)
	}
	biasRows, _ := ptBias.Dims()
	newCiphertexts := make([]*rlwe.Ciphertext, biasRows)
	ctRealRow := modelParams.NumRealRow
	ctBatch := modelParams.NumBatch
	fmt.Print("ctRealRow:", ctRealRow, " ctBatch:", ctBatch, " biasRows:", biasRows, "\n")
	// fmt.Println(ctRealRow * ctBatch)
	for i := 0; i < biasRows; i++ {
		biasVal := ptBias.At(i, 0)

		biasVec := make([]float64, ctBatch*ctRealRow)
		for j := 0; j < ctBatch*ctRealRow; j++ {
			biasVec[j] = biasVal
		}
		ct, err := eval.AddNew(ctMul.Ciphertexts[i], biasVec)
		if err != nil {
			return nil, err
		}
		newCiphertexts[i] = ct
	}

	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ctMul.NumBatch,
		NumRow:      ctMul.NumRow,
		NumCol:      biasRows,
	}, nil
}

/*
 * RotateCiphertextMatrices 对 CiphertextMatrices 中所有密文进行指定步长的旋转
 * 输入: ctMatrices *he.CiphertextMatrices, modelParams *configs.ModelParams, steps int, eval *ckks.Evaluator
 * 输出: *he.CiphertextMatrices，所有密文均已旋转
 */
func RotateCiphertextMatrices(ctMatrices *he.CiphertextMatrices, steps int, eval *ckks.Evaluator) *he.CiphertextMatrices {
	newCiphertexts := make([]*rlwe.Ciphertext, len(ctMatrices.Ciphertexts))
	var err error
	for i, ct := range ctMatrices.Ciphertexts {
		newCiphertexts[i], err = eval.RotateNew(ct, steps)
		if err != nil {
			panic(err)
		}
	}
	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ctMatrices.NumBatch,
		NumRow:      ctMatrices.NumRow,
		NumCol:      ctMatrices.NumCol,
	}
}

/*
 * RotateCiphertextMatricesHoisting
 * 对 CiphertextMatrices 中所有密文，依次按 stepsSlice 中每个步长进行批量旋转（Hoisted），
 * 利用 RotateHoistedNew 一次性高效计算所有步长的旋转结果。
 * 输入: ctMatrices *he.CiphertextMatrices, stepsSlice []int, eval *ckks.Evaluator
 * 输出: []*he.CiphertextMatrices，长度等于 stepsSlice，每个元素为对应步长旋转后的 CiphertextMatrices
 * 说明: RotateHoistedNew 会返回一个 map[int]*Ciphertext，键为步长，值为旋转后的密文。
 *      本函数对每个密文调用一次 RotateHoistedNew，然后按步长收集所有密文，组装为 CiphertextMatrices 切片。
 */
func RotateCiphertextMatricesHoisting(
	ctMatrices *he.CiphertextMatrices,
	stepsSlice []int,
	eval *ckks.Evaluator,
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
	// 对每个密文，批量旋转所有步长
	for colIdx, ct := range ctMatrices.Ciphertexts {
		rotMap, err := eval.RotateHoistedNew(ct, stepsSlice)
		if err != nil {
			panic(err)
		}
		for stepIdx, step := range stepsSlice {
			result[stepIdx].Ciphertexts[colIdx] = rotMap[step]
		}
	}
	return result
}

// SplitCiphertextMatricesByHeads
// 将一个 *he.CiphertextMatrices 按照 modelParams.NumHeads 切分成多个 *he.CiphertextMatrices（每个包含等量列）
// 假设 NumCol 能被 NumHeads 整除
func SplitCiphertextMatricesByHeads(ctMatrices *he.CiphertextMatrices, modelParams *configs.ModelParams) []*he.CiphertextMatrices {
	numHeads := modelParams.NumHeads
	colsPerHead := ctMatrices.NumCol / numHeads
	result := make([]*he.CiphertextMatrices, numHeads)
	for h := 0; h < numHeads; h++ {
		start := h * colsPerHead
		end := (h + 1) * colsPerHead
		result[h] = &he.CiphertextMatrices{
			Ciphertexts: ctMatrices.Ciphertexts[start:end],
			NumBatch:    ctMatrices.NumBatch,
			NumRow:      ctMatrices.NumRow,
			NumCol:      colsPerHead,
		}
	}
	return result
}

// SplitCiphertextMatricesByHeads
// 将一个 *he.CiphertextMatrices 按照 modelParams.NumHeads 切分成多个 *he.CiphertextMatrices（每个包含等量列）
// 假设 NumCol 能被 NumHeads 整除
func SplitCiphertextMatricesByNumber(ctMatrices *he.CiphertextMatrices, numHeads int) []*he.CiphertextMatrices {
	colsPerHead := ctMatrices.NumCol / numHeads
	result := make([]*he.CiphertextMatrices, numHeads)
	for h := 0; h < numHeads; h++ {
		start := h * colsPerHead
		end := (h + 1) * colsPerHead
		result[h] = &he.CiphertextMatrices{
			Ciphertexts: ctMatrices.Ciphertexts[start:end],
			NumBatch:    ctMatrices.NumBatch,
			NumRow:      ctMatrices.NumRow,
			NumCol:      colsPerHead,
		}
	}
	return result
}

// MergeCiphertextMatricesByHeads
// 将多个 *he.CiphertextMatrices 合并成一个（按列拼接）
// 要求所有输入的 NumBatch、NumRow 相同
func MergeCiphertextMatricesByHeads(mats []*he.CiphertextMatrices) *he.CiphertextMatrices {
	if len(mats) == 0 {
		return &he.CiphertextMatrices{}
	}
	numBatch := mats[0].NumBatch
	numRow := mats[0].NumRow
	totalCols := 0
	for _, m := range mats {
		totalCols += m.NumCol
	}
	allCts := make([]*rlwe.Ciphertext, 0, totalCols)
	for _, m := range mats {
		allCts = append(allCts, m.Ciphertexts...)
	}
	return &he.CiphertextMatrices{
		Ciphertexts: allCts,
		NumBatch:    numBatch,
		NumRow:      numRow,
		NumCol:      totalCols,
	}
}

/*
 * CiphertextMatricesMultiplyCiphertextMatricesToHaleviShoup
 * 输入:  ctQ, ctK *he.CiphertextMatrices, modelParams *configs.ModelParams, ckksParams ckks.Parameters, eval *ckks.Evaluator
 * 输出: *he.CiphertextMatrices, error
 * 计算: ctQ(a, b, c) × [ctK(a, b, c)^T --> ctK^T(a, c, b)] --> ctQK(a, b, b)
 * 说明: 使用大步-小步（giant-step/baby-step）策略和 hoisting 技术高效完成 Halevi-Shoup 编码的密文矩阵乘法。
 *       计算目标 Q × K^T
 */
func CiphertextMatricesMultiplyCiphertextMatricesToHaleviShoup(
	ctQ, ctK *he.CiphertextMatrices,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	eval *ckks.Evaluator,
) (*he.CiphertextMatrices, error) {
	ctRows := ctQ.NumRow
	ctCols := ctQ.NumCol
	ctBatch := ctQ.NumBatch

	babyStep := modelParams.BabyStep
	giantStep := modelParams.GiantStep
	baseLen := modelParams.NumBatch
	var err error

	// 检查维度
	if ctQ.NumBatch != ctK.NumBatch ||
		ctQ.NumRow != ctK.NumRow ||
		ctQ.NumCol != ctK.NumCol {
		return &he.CiphertextMatrices{}, fmt.Errorf("can not multiply ctQ and ctK transpose to Halevi-Shoup encoding")
	}

	// 生成所有步长
	babySteps := make([]int, 0, babyStep)
	for i := 0; i < babyStep; i++ {
		babySteps = append(babySteps, i*baseLen)
	}
	giantSteps := make([]int, 0, giantStep)
	for i := 0; i < giantStep; i++ {
		giantSteps = append(giantSteps, -i*baseLen*babyStep)
	}

	// 使用 hoisting 技术批量旋转所有的密文矩阵
	ctQRotated := RotateCiphertextMatricesHoisting(ctQ, giantSteps, eval)
	ctKRotated := RotateCiphertextMatricesHoisting(ctK, babySteps, eval)

	// 进行密文乘法
	newCiphertexts := make([]*rlwe.Ciphertext, ctRows)
	for i := 0; i < giantStep; i++ {

		for j := 0; j < babyStep; j++ {
			// 计算 ctQRotated[i] 和 ctKRotated[j] 的乘积
			if i*babyStep+j >= ctRows {
				break // 超出行数，退出循环
			}
			//用于存储密文
			ct := ckks.NewCiphertext(ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
			for k := 0; k < ctCols; k++ {
				err = eval.MulRelinThenAdd(ctQRotated[i].Ciphertexts[k], ctKRotated[j].Ciphertexts[k], ct)
				if err != nil {
					panic(err)
				}
			}
			if err = eval.Rescale(ct, ct); err != nil {
				panic(err)
			}
			newCiphertexts[i*babyStep+j] = ct
		}
	}

	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ctBatch,
		NumRow:      ctRows,
		NumCol:      ctCols,
	}, nil
}

/*
 * CiphertextMatricesComputeAttentionWithBSGS
 * 功能：
 *   对三个密文矩阵 ctQ、ctK、ctV，先用 Halevi-Shoup（大步-小步+hoisting）策略计算 Q × K^T，
 *   然后对每个乘积结果减去 modelParams.ExpSubValue，并对每个元素做指数函数（exp）近似（如Chebyshev多项式），
 *   再与V做乘法，最后对每一行的exp结果进行求和，返回所有exp结果的密文矩阵和每行和的密文。
 *
 * 输入参数：
 *   ctQ, ctK, ctV - 输入的密文矩阵（*he.CiphertextMatrices），维度需一致
 *   modelParams   - 模型参数（*configs.ModelParams），包含大步/小步参数和exp偏移
 *   ckksParams    - CKKS参数
 *   eval          - CKKS Evaluator
 *
 * 输出参数：
 *   *he.CiphertextMatrices - 经过exp处理并与V相乘后的密文矩阵（即attention输出）
 *   *rlwe.Ciphertext       - 每行exp和的密文（可用于softmax分母）
 *   error                  - 错误信息
 *
 */
func CiphertextMatricesComputeAttentionWithBSGS(
	ctQ, ctK, ctV *he.CiphertextMatrices,
	LayerNumber int,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	eval *ckks.Evaluator,
) (*he.CiphertextMatrices, *rlwe.Ciphertext, error) {
	ctRows := ctQ.NumRow
	ctCols := ctQ.NumCol
	ctBatch := ctQ.NumBatch

	babyStep := modelParams.BabyStep
	giantStep := modelParams.GiantStep
	baseLen := modelParams.NumBatch

	// 检查QKV是否一致
	if ctQ.NumBatch != ctK.NumBatch || ctQ.NumBatch != ctV.NumBatch {
		return nil, nil, fmt.Errorf("ctQ, ctK and ctV must have the same Batch: ctQ:%d , ctK:%d , ctV: %d", ctQ.NumBatch, ctK.NumBatch, ctV.NumBatch)
	}
	if ctQ.NumRow != ctK.NumRow || ctQ.NumRow != ctV.NumRow {
		return nil, nil, fmt.Errorf("ctQ, ctK and ctV must have the same Row: ctQ:%d , ctK:%d , ctV: %d", ctQ.NumRow, ctK.NumRow, ctV.NumRow)
	}
	if ctQ.NumCol != ctK.NumCol || ctQ.NumCol != ctV.NumCol {
		return nil, nil, fmt.Errorf("ctQ, ctK and ctV must have the same Col: ctQ:%d , ctK:%d , ctV: %d", ctQ.NumCol, ctK.NumCol, ctV.NumCol)
	}

	// 生成所有步长
	babySteps := make([]int, 0, babyStep)
	for i := 0; i < babyStep; i++ {
		babySteps = append(babySteps, i*baseLen)
	}
	giantSteps := make([]int, 0, giantStep)
	for i := 0; i < giantStep; i++ {
		giantSteps = append(giantSteps, -i*baseLen*babyStep)
	}

	// 使用 hoisting 技术批量旋转所有的密文矩阵
	ctQRotated := RotateCiphertextMatricesHoisting(ctQ, giantSteps, eval)
	ctKRotated := RotateCiphertextMatricesHoisting(ctK, babySteps, eval)
	ctVRotated := RotateCiphertextMatricesHoisting(ctV, babySteps, eval)

	// 进行密文乘法
	newCiphertexts := make([]*rlwe.Ciphertext, ctCols)
	for k := 0; k < ctCols; k++ {
		newCiphertexts[k] = ckks.NewCiphertext(ckksParams, ctV.Ciphertexts[0].Degree(), ctV.Ciphertexts[0].Level())
	}
	ctAddAllbyRow := ckks.NewCiphertext(ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
	for i := 0; i < giantStep; i++ {

		// 用于存储每个giantStep的结果
		ctAddGaintStep := ckks.NewCiphertext(ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
		// 这里的ctQKV是一个新的CiphertextMatrices，用于存储 QKV 的结果
		localNewCiphertexts := make([]*rlwe.Ciphertext, ctV.NumCol)
		for k := 0; k < ctV.NumCol; k++ {
			localNewCiphertexts[k] = ckks.NewCiphertext(ckksParams, ctV.Ciphertexts[0].Degree(), ctV.Ciphertexts[0].Level())
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

			// step1: 计算 ctQRotated[i] 和 ctKRotated[j] 的乘积
			ct, err := CiphertextMatricesMultiplyCiphertextMatricesThenAdd(ctQRotated[i], ctKRotated[j], &ckksParams, eval)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to multiply ctQRotated[%d] and ctKRotated[%d]: %w", i, j, err)
			}

			// step2: 计算Exp(ct - modelParams.ExpSubValue)
			err = eval.Sub(ct, modelParams.ExpSubValue[LayerNumber], ct)
			if err != nil {
				panic(err)
			}
			ctExp := softmax.CiphertextExpChebyshev(ct, &ckksParams, eval, modelParams.ExpMinValue, modelParams.ExpMaxValue, modelParams.ExpDegree)

			// step3: 将Exp(ct - modelParams.ExpSubValue) * ctVRotated[j] 存入ctQKV
			err = CiphertextMatricesMultiplyCiphertextAddToRes(ctVRotated[j], ctExp, ctQKV, &ckksParams, eval)
			if err != nil {
				panic(err)

			}

			// 用于求和SUM(Exp(ct - modelParams.ExpSubValue))
			err = eval.Add(ctAddGaintStep, ctExp, ctAddGaintStep)
			if err != nil {
				panic(err)
			}
		}
		// 旋转QKV的结果
		QKVRotKi := RotateCiphertextMatrices(ctQKV, i*baseLen*babyStep, eval)
		// 将QKVRotKi的结果累加到newCiphertexts中
		for k := 0; k < ctCols; k++ {
			err := eval.Add(newCiphertexts[k], QKVRotKi.Ciphertexts[k], newCiphertexts[k])
			if err != nil {
				panic(err)
			}
		}

		// 用于求和SUM(Exp(ct - modelParams.ExpSubValue))
		ctAddRotate, err := eval.RotateNew(ctAddGaintStep, i*baseLen*babyStep)
		if err != nil {
			panic(err)
		}
		err = eval.Add(ctAddAllbyRow, ctAddRotate, ctAddAllbyRow)
		if err != nil {
			panic(err)
		}
	}

	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ctBatch,
		NumRow:      ctRows,
		NumCol:      ctCols,
	}, ctAddAllbyRow, nil
}

/*
 * CiphertextMatricesComputeAttentionWithBSGSAndApproxMax
 * 功能：
 *   对三个密文矩阵 ctQ、ctK、ctV，先用 Halevi-Shoup（大步-小步+hoisting）策略计算 Q × K^T，
 *   然后对每个乘积结果减去 modelParams.ExpSubValue，并对每个元素做指数函数（exp）近似（如Chebyshev多项式），
 *   再与V做乘法，最后对每一行的exp结果进行求和，返回所有exp结果的密文矩阵和每行和的密文。
 *   在计算时，最大值不用固定值，而使用近似值进行计算
 *   均值：rowMean  方差：rowVar  标准差：rowStd = 1.25 + 0.1 * rowVar
 *   近似最大值：rowMean + rowStd * 2.08
 *   MASK_Attention以01向量掩盖（这里，mask使用其他方式隐藏起来）
 *   在计算方差时，通过修改公式将排除0元素对方差的影响
 *
 * 输入参数：
 *   ctQ, ctK, ctV - 输入的密文矩阵（*he.CiphertextMatrices），维度需一致
 *   modelParams   - 模型参数（*configs.ModelParams），包含大步/小步参数和exp偏移
 *   ckksParams    - CKKS参数
 *   eval          - CKKS Evaluator
 *
 * 输出参数：
 *   *he.CiphertextMatrices - 经过exp处理并与V相乘后的密文矩阵（即attention输出）
 *   *rlwe.Ciphertext       - 每行exp和的密文（可用于softmax分母）
 *   error                  - 错误信息
 *
 */

func CiphertextMatricesComputeAttentionWithBSGSAndApproxMax(
	ctQ, ctK, ctV *he.CiphertextMatrices,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	eval *ckks.Evaluator,
) (*he.CiphertextMatrices, *rlwe.Ciphertext, error) {
	ctRows := ctQ.NumRow
	ctCols := ctQ.NumCol
	ctBatch := ctQ.NumBatch

	babyStep := modelParams.BabyStep
	giantStep := modelParams.GiantStep
	baseLen := modelParams.NumBatch
	realRow := modelParams.NumRealRow

	// 生成掩码矩阵
	maskMat := utils.GenerateSpecialSquareMatrix(modelParams)

	// 检查QKV是否一致
	if ctQ.NumBatch != ctK.NumBatch || ctQ.NumBatch != ctV.NumBatch {
		return nil, nil, fmt.Errorf("ctQ, ctK and ctV must have the same Batch: ctQ:%d , ctK:%d , ctV: %d", ctQ.NumBatch, ctK.NumBatch, ctV.NumBatch)
	}
	if ctQ.NumRow != ctK.NumRow || ctQ.NumRow != ctV.NumRow {
		return nil, nil, fmt.Errorf("ctQ, ctK and ctV must have the same Row: ctQ:%d , ctK:%d , ctV: %d", ctQ.NumRow, ctK.NumRow, ctV.NumRow)
	}
	if ctQ.NumCol != ctK.NumCol || ctQ.NumCol != ctV.NumCol {
		return nil, nil, fmt.Errorf("ctQ, ctK and ctV must have the same Col: ctQ:%d , ctK:%d , ctV: %d", ctQ.NumCol, ctK.NumCol, ctV.NumCol)
	}

	// fmt.Println(babyStep, giantStep)
	// 生成所有步长
	babySteps := make([]int, 0, babyStep)
	for i := 0; i < babyStep; i++ {
		babySteps = append(babySteps, i*baseLen)
	}
	giantSteps := make([]int, 0, giantStep)
	for i := 0; i < giantStep; i++ {
		giantSteps = append(giantSteps, -i*baseLen*babyStep)
	}

	// 使用 hoisting 技术批量旋转所有的密文矩阵
	ctQRotated := RotateCiphertextMatricesHoisting(ctQ, giantSteps, eval)
	ctKRotated := RotateCiphertextMatricesHoisting(ctK, babySteps, eval)
	ctVRotated := RotateCiphertextMatricesHoisting(ctV, babySteps, eval)

	ctQKTCiphertext := make([]*rlwe.Ciphertext, ctRows)
	ctQKTAdd := ckks.NewCiphertext(ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
	// Step1. Compute Add All Row
	for i := 0; i < giantStep; i++ {
		// 用于存储每个giantStep的结果
		ctAddGaintStep := ckks.NewCiphertext(ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
		for j := 0; j < babyStep; j++ {
			// 检查是否超出行数
			if i*babyStep+j >= ctRows {
				break // 超出行数，退出循环
			}

			// step1.1: 计算 ctQRotated[i] 和 ctKRotated[j] 的乘积
			ct, err := CiphertextMatricesMultiplyCiphertextMatricesThenAdd(ctQRotated[i], ctKRotated[j], &ckksParams, eval)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to multiply ctQRotated[%d] and ctKRotated[%d]: %w", i, j, err)
			}

			// step1.2:用于求和SUM(ctQKT)
			err = eval.Add(ctAddGaintStep, ct, ctAddGaintStep)
			if err != nil {
				panic(err)
			}
			ctQKTCiphertext[i*babyStep+j] = ct
		}
		// 用于求和SUM(ctQKT)
		ctAddRotate, err := eval.RotateNew(ctAddGaintStep, i*baseLen*babyStep)
		if err != nil {
			panic(err)
		}
		err = eval.Add(ctQKTAdd, ctAddRotate, ctQKTAdd)
		if err != nil {
			panic(err)
		}
	}

	// Step2. Compute Approx Max
	ctMean, err := eval.MulRelinNew(ctQKTAdd, 1/float64(realRow))
	if err != nil {
		panic(err)
	}
	if err = eval.Rescale(ctMean, ctMean); err != nil {
		panic(err)
	}
	ctQKTAddRot, err := eval.RotateHoistedNew(ctQKTAdd, giantSteps)
	if err != nil {
		panic(err)
	}

	// 计算scale的系数
	// constantValue := 0.944 * math.Sqrt(float64(modelParams.NumRow))
	constantValue := modelParams.ConstantValue
	varScale := math.Sqrt(0.1*(1.0/float64(realRow))*constantValue) * (1.0 / float64(realRow))
	constValue := 1.25 * constantValue
	// fmt.Println(varScale, "  ", constValue)

	ctVarScale := ckks.NewCiphertext(ckksParams, ctMean.Degree(), ctMean.Level())
	for i := 0; i < giantStep; i++ {
		// 用于存储每个giantStep的结果
		ctAddGaintStep := ckks.NewCiphertext(ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
		for j := 0; j < babyStep; j++ {
			// 检查是否超出行数
			if i*babyStep+j >= ctRows {
				break // 超出行数，退出循环
			}
			// compute d * x_i - SUM
			ctQKTScale, err := eval.MulNew(ctQKTCiphertext[i*babyStep+j], realRow)
			if err != nil {
				panic(err)
			}
			// Sub SUM Value
			ctSub, err := eval.SubNew(ctQKTScale, ctQKTAddRot[-i*baseLen*babyStep])
			if err != nil {
				panic(err)
			}

			maskVec := utils.ExtractAndRepeatDiagonal(maskMat, i*babyStep+j, modelParams.NumBatch)
			for idx := range maskVec {
				maskVec[idx] *= varScale
			}
			maskVecRot := utils.RotateSliceNew(maskVec, -i*baseLen*babyStep)

			if err = eval.MulRelin(ctSub, maskVecRot, ctSub); err != nil {
				panic(err)
			}
			if err = eval.Rescale(ctSub, ctSub); err != nil {
				panic(err)
			}

			// Sub Mean Valuen Square
			if err = eval.MulRelin(ctSub, ctSub, ctSub); err != nil {
				panic(err)
			}
			if err = eval.Rescale(ctSub, ctSub); err != nil {
				panic(err)
			}

			// accumulate add gaint value
			err = eval.Add(ctAddGaintStep, ctSub, ctAddGaintStep)
			if err != nil {
				panic(err)
			}
		}
		// 用于求和SUM(ctQKT)
		ctAddRotate, err := eval.RotateNew(ctAddGaintStep, i*baseLen*babyStep)
		if err != nil {
			panic(err)
		}
		err = eval.Add(ctVarScale, ctAddRotate, ctVarScale)
		if err != nil {
			panic(err)
		}
	}

	// approx std
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

	// Step3. Sub approx max and compute QK
	newCiphertexts := make([]*rlwe.Ciphertext, ctCols)
	for k := 0; k < ctCols; k++ {
		newCiphertexts[k] = ckks.NewCiphertext(ckksParams, ctApproxMax.Degree(), ctApproxMax.Level())
	}
	ctAddAllbyRow := ckks.NewCiphertext(ckksParams, ctApproxMax.Degree(), ctApproxMax.Level())
	for i := 0; i < giantStep; i++ {

		// 用于存储每个giantStep的结果
		ctAddGaintStep := ckks.NewCiphertext(ckksParams, ctApproxMax.Degree(), ctApproxMax.Level())
		// 这里的ctQKV是一个新的CiphertextMatrices，用于存储 QKV 的结果
		localNewCiphertexts := make([]*rlwe.Ciphertext, ctV.NumCol)
		for k := 0; k < ctV.NumCol; k++ {
			localNewCiphertexts[k] = ckks.NewCiphertext(ckksParams, ctApproxMax.Degree(), ctApproxMax.Level())
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

			// step1: 计算 减去最大值
			ct, err := eval.SubNew(ctQKTCiphertext[i*babyStep+j], ctApproxMaxRot[-i*baseLen*babyStep])
			if err != nil {
				panic(err)
			}
			// step2: 计算exp函数
			ctExp := softmax.CiphertextExpChebyshev(ct, &ckksParams, eval, modelParams.ExpMinValue, modelParams.ExpMaxValue, modelParams.ExpDegree)

			// step3：计算乘以mask attention
			maskVec := utils.ExtractAndRepeatDiagonal(maskMat, i*babyStep+j, modelParams.NumBatch)
			maskVecRot := utils.RotateSliceNew(maskVec, -i*baseLen*babyStep)
			if err = eval.MulRelin(ctExp, maskVecRot, ctExp); err != nil {
				panic(err)
			}
			if err = eval.Rescale(ctExp, ctExp); err != nil {
				panic(err)
			}

			// step4: 将Exp(ct - ctApproxMax) * ctVRotated[j] 存入ctQKV
			err = CiphertextMatricesMultiplyCiphertextAddToRes(ctVRotated[j], ctExp, ctQKV, &ckksParams, eval)
			if err != nil {
				panic(err)

			}

			// 用于求和SUM(Exp(ct - modelParams.ExpSubValue))
			err = eval.Add(ctAddGaintStep, ctExp, ctAddGaintStep)
			if err != nil {
				panic(err)
			}
		}
		// 旋转QKV的结果
		QKVRotKi := RotateCiphertextMatrices(ctQKV, i*baseLen*babyStep, eval)
		// 将QKVRotKi的结果累加到newCiphertexts中
		for k := 0; k < ctCols; k++ {
			err := eval.Add(newCiphertexts[k], QKVRotKi.Ciphertexts[k], newCiphertexts[k])
			if err != nil {
				panic(err)
			}
		}

		// 用于求和SUM(Exp(ct - modelParams.ExpSubValue))
		ctAddRotate, err := eval.RotateNew(ctAddGaintStep, i*baseLen*babyStep)
		if err != nil {
			panic(err)
		}
		err = eval.Add(ctAddAllbyRow, ctAddRotate, ctAddAllbyRow)
		if err != nil {
			panic(err)
		}
	}

	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ctBatch,
		NumRow:      ctRows,
		NumCol:      ctCols,
	}, ctAddAllbyRow, nil
}

/*
 * CiphertextMatricesComputeAttentionWithBSGSAndApproxMax
 * 功能：
 *   对三个密文矩阵 ctQ、ctK、ctV，先用 Halevi-Shoup（大步-小步+hoisting）策略计算 Q × K^T，
 *   然后对每个乘积结果减去 modelParams.ExpSubValue，并对每个元素做指数函数（exp）近似（如Chebyshev多项式），
 *   再与V做乘法，最后对每一行的exp结果进行求和，返回所有exp结果的密文矩阵和每行和的密文。
 *   在计算时，最大值不用固定值，而使用近似值进行计算
 *   均值：rowMean  方差：rowVar  标准差：rowStd = 1.25 + 0.1 * rowVar
 *   近似最大值：rowMean + rowStd * 2.08
 *   MASK_Attention以01向量掩盖（这里，mask使用其他方式隐藏起来）
 *   在计算方差时，通过修改公式将排除0元素对方差的影响
 *
 * 输入参数：
 *   ctQ, ctK, ctV - 输入的密文矩阵（*he.CiphertextMatrices），维度需一致
 *   modelParams   - 模型参数（*configs.ModelParams），包含大步/小步参数和exp偏移
 *   ckksParams    - CKKS参数
 *   eval          - CKKS Evaluator
 *
 * 输出参数：
 *   *he.CiphertextMatrices - 经过exp处理并与V相乘后的密文矩阵（即attention输出）
 *   *rlwe.Ciphertext       - 每行exp和的密文（可用于softmax分母）
 *   error                  - 错误信息
 *
 */

func GroupCiphertextMatricesComputeAttentionWithBSGSAndApproxMax(
	ctQ *he.CiphertextMatrices,
	ctKRotated, ctVRotated []*he.CiphertextMatrices,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	eval *ckks.Evaluator,
) (*he.CiphertextMatrices, *rlwe.Ciphertext, error) {
	ctRows := ctQ.NumRow
	ctCols := ctQ.NumCol
	ctBatch := ctQ.NumBatch

	babyStep := modelParams.BabyStep
	giantStep := modelParams.GiantStep
	baseLen := modelParams.NumBatch
	realRow := modelParams.NumRealRow

	// 生成掩码矩阵
	maskMat := utils.GenerateSpecialSquareMatrixLlama(modelParams)

	// 检查QKV是否一致
	if ctQ.NumBatch != ctKRotated[0].NumBatch || ctQ.NumBatch != ctVRotated[0].NumBatch {
		return nil, nil, fmt.Errorf("ctQ, ctK and ctV must have the same Batch: ctQ:%d , ctK:%d , ctV: %d", ctQ.NumBatch, ctKRotated[0].NumBatch, ctVRotated[0].NumBatch)
	}
	if ctQ.NumRow != ctKRotated[0].NumRow || ctQ.NumRow != ctVRotated[0].NumRow {
		return nil, nil, fmt.Errorf("ctQ, ctK and ctV must have the same Row: ctQ:%d , ctK:%d , ctV: %d", ctQ.NumRow, ctKRotated[0].NumRow, ctVRotated[0].NumRow)
	}
	if ctQ.NumCol != ctKRotated[0].NumCol || ctQ.NumCol != ctVRotated[0].NumCol {
		return nil, nil, fmt.Errorf("ctQ, ctK and ctV must have the same Col: ctQ:%d , ctK:%d , ctV: %d", ctQ.NumCol, ctKRotated[0].NumCol, ctVRotated[0].NumCol)
	}

	// fmt.Println(babyStep, giantStep)
	// 生成所有步长
	babySteps := make([]int, 0, babyStep)
	for i := 0; i < babyStep; i++ {
		babySteps = append(babySteps, i*baseLen)
	}
	giantSteps := make([]int, 0, giantStep)
	for i := 0; i < giantStep; i++ {
		giantSteps = append(giantSteps, -i*baseLen*babyStep)
	}

	// 使用 hoisting 技术批量旋转所有的密文矩阵
	ctQRotated := RotateCiphertextMatricesHoisting(ctQ, giantSteps, eval)

	ctQKTCiphertext := make([]*rlwe.Ciphertext, ctRows)
	ctQKTAdd := ckks.NewCiphertext(ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
	// Step1. Compute Add All Row
	for i := 0; i < giantStep; i++ {
		// 用于存储每个giantStep的结果
		ctAddGaintStep := ckks.NewCiphertext(ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
		for j := 0; j < babyStep; j++ {
			// 检查是否超出行数
			if i*babyStep+j >= ctRows {
				break // 超出行数，退出循环
			}

			// step1.1: 计算 ctQRotated[i] 和 ctKRotated[j] 的乘积
			ct, err := CiphertextMatricesMultiplyCiphertextMatricesThenAdd(ctQRotated[i], ctKRotated[j], &ckksParams, eval)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to multiply ctQRotated[%d] and ctKRotated[%d]: %w", i, j, err)
			}

			// step1.2:用于求和SUM(ctQKT)
			err = eval.Add(ctAddGaintStep, ct, ctAddGaintStep)
			if err != nil {
				panic(err)
			}
			ctQKTCiphertext[i*babyStep+j] = ct
		}
		// 用于求和SUM(ctQKT)
		ctAddRotate, err := eval.RotateNew(ctAddGaintStep, i*baseLen*babyStep)
		if err != nil {
			panic(err)
		}
		err = eval.Add(ctQKTAdd, ctAddRotate, ctQKTAdd)
		if err != nil {
			panic(err)
		}
	}

	// Step2. Compute Approx Max
	ctMean, err := eval.MulRelinNew(ctQKTAdd, 1/float64(realRow))
	if err != nil {
		panic(err)
	}
	if err = eval.Rescale(ctMean, ctMean); err != nil {
		panic(err)
	}
	ctQKTAddRot, err := eval.RotateHoistedNew(ctQKTAdd, giantSteps)
	if err != nil {
		panic(err)
	}

	// 计算scale的系数
	// constantValue := 0.944 * math.Sqrt(float64(modelParams.NumRow))
	constantValue := modelParams.ConstantValue
	varScale := math.Sqrt(0.1*(1.0/float64(realRow))*constantValue) * (1.0 / float64(realRow))
	constValue := 1.25 * constantValue
	// fmt.Println(varScale, "  ", constValue)

	ctVarScale := ckks.NewCiphertext(ckksParams, ctMean.Degree(), ctMean.Level())
	for i := 0; i < giantStep; i++ {
		// 用于存储每个giantStep的结果
		ctAddGaintStep := ckks.NewCiphertext(ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
		for j := 0; j < babyStep; j++ {
			// 检查是否超出行数
			if i*babyStep+j >= ctRows {
				break // 超出行数，退出循环
			}
			// compute d * x_i - SUM
			ctQKTScale, err := eval.MulNew(ctQKTCiphertext[i*babyStep+j], realRow)
			if err != nil {
				panic(err)
			}
			// Sub SUM Value
			ctSub, err := eval.SubNew(ctQKTScale, ctQKTAddRot[-i*baseLen*babyStep])
			if err != nil {
				panic(err)
			}

			maskVec := utils.ExtractAndRepeatDiagonal(maskMat, i*babyStep+j, modelParams.NumBatch)
			for idx := range maskVec {
				maskVec[idx] *= varScale
			}
			maskVecRot := utils.RotateSliceNew(maskVec, -i*baseLen*babyStep)

			if err = eval.MulRelin(ctSub, maskVecRot, ctSub); err != nil {
				panic(err)
			}
			if err = eval.Rescale(ctSub, ctSub); err != nil {
				panic(err)
			}

			// Sub Mean Valuen Square
			if err = eval.MulRelin(ctSub, ctSub, ctSub); err != nil {
				panic(err)
			}
			if err = eval.Rescale(ctSub, ctSub); err != nil {
				panic(err)
			}

			// accumulate add gaint value
			err = eval.Add(ctAddGaintStep, ctSub, ctAddGaintStep)
			if err != nil {
				panic(err)
			}
		}
		// 用于求和SUM(ctQKT)
		ctAddRotate, err := eval.RotateNew(ctAddGaintStep, i*baseLen*babyStep)
		if err != nil {
			panic(err)
		}
		err = eval.Add(ctVarScale, ctAddRotate, ctVarScale)
		if err != nil {
			panic(err)
		}
	}

	// approx std
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

	// Step3. Sub approx max and compute QK
	newCiphertexts := make([]*rlwe.Ciphertext, ctCols)
	for k := 0; k < ctCols; k++ {
		newCiphertexts[k] = ckks.NewCiphertext(ckksParams, ctApproxMax.Degree(), ctApproxMax.Level())
	}
	ctAddAllbyRow := ckks.NewCiphertext(ckksParams, ctApproxMax.Degree(), ctApproxMax.Level())
	for i := 0; i < giantStep; i++ {

		// 用于存储每个giantStep的结果
		ctAddGaintStep := ckks.NewCiphertext(ckksParams, ctApproxMax.Degree(), ctApproxMax.Level())
		// 这里的ctQKV是一个新的CiphertextMatrices，用于存储 QKV 的结果
		localNewCiphertexts := make([]*rlwe.Ciphertext, ctVRotated[0].NumCol)
		for k := 0; k < ctVRotated[0].NumCol; k++ {
			localNewCiphertexts[k] = ckks.NewCiphertext(ckksParams, ctApproxMax.Degree(), ctApproxMax.Level())
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

			// step1: 计算 减去最大值
			ct, err := eval.SubNew(ctQKTCiphertext[i*babyStep+j], ctApproxMaxRot[-i*baseLen*babyStep])
			if err != nil {
				panic(err)
			}
			// step2: 计算exp函数
			ctExp := softmax.CiphertextExpChebyshev(ct, &ckksParams, eval, modelParams.ExpMinValue, modelParams.ExpMaxValue, modelParams.ExpDegree)

			// step3：计算乘以mask attention
			maskVec := utils.ExtractAndRepeatDiagonal(maskMat, i*babyStep+j, modelParams.NumBatch)
			maskVecRot := utils.RotateSliceNew(maskVec, -i*baseLen*babyStep)
			if err = eval.MulRelin(ctExp, maskVecRot, ctExp); err != nil {
				panic(err)
			}
			if err = eval.Rescale(ctExp, ctExp); err != nil {
				panic(err)
			}

			// step4: 将Exp(ct - ctApproxMax) * ctVRotated[j] 存入ctQKV
			err = CiphertextMatricesMultiplyCiphertextAddToRes(ctVRotated[j], ctExp, ctQKV, &ckksParams, eval)
			if err != nil {
				panic(err)

			}

			// 用于求和SUM(Exp(ct - modelParams.ExpSubValue))
			err = eval.Add(ctAddGaintStep, ctExp, ctAddGaintStep)
			if err != nil {
				panic(err)
			}
		}
		// 旋转QKV的结果
		QKVRotKi := RotateCiphertextMatrices(ctQKV, i*baseLen*babyStep, eval)
		// 将QKVRotKi的结果累加到newCiphertexts中
		for k := 0; k < ctCols; k++ {
			err := eval.Add(newCiphertexts[k], QKVRotKi.Ciphertexts[k], newCiphertexts[k])
			if err != nil {
				panic(err)
			}
		}

		// 用于求和SUM(Exp(ct - modelParams.ExpSubValue))
		ctAddRotate, err := eval.RotateNew(ctAddGaintStep, i*baseLen*babyStep)
		if err != nil {
			panic(err)
		}
		err = eval.Add(ctAddAllbyRow, ctAddRotate, ctAddAllbyRow)
		if err != nil {
			panic(err)
		}
	}

	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ctBatch,
		NumRow:      ctRows,
		NumCol:      ctCols,
	}, ctAddAllbyRow, nil
}

/*
 * CiphertextMatricesComputeAttentionWithBSGSAndApproxMax
 * 功能：
 *   对三个密文矩阵 ctQ、ctK、ctV，先用 Halevi-Shoup（大步-小步+hoisting）策略计算 Q × K^T，
 *   然后对每个乘积结果减去 modelParams.ExpSubValue，并对每个元素做指数函数（exp）近似（如Chebyshev多项式），
 *   再与V做乘法，最后对每一行的exp结果进行求和，返回所有exp结果的密文矩阵和每行和的密文。
 *   在计算时，最大值不用固定值，而使用近似值进行计算
 *   均值：rowMean  方差：rowVar  标准差：rowStd = 1.25 + 0.1 * rowVar
 *   近似最大值：rowMean + rowStd * sqrt(2 * log(RealRow))
 *   MASK_Attention以最小值掩盖
 *
 * 输入参数：
 *   ctQ, ctK, ctV - 输入的密文矩阵（*he.CiphertextMatrices），维度需一致
 *   modelParams   - 模型参数（*configs.ModelParams），包含大步/小步参数和exp偏移
 *   ckksParams    - CKKS参数
 *   eval          - CKKS Evaluator
 *
 * 输出参数：
 *   *he.CiphertextMatrices - 经过exp处理并与V相乘后的密文矩阵（即attention输出）
 *   *rlwe.Ciphertext       - 每行exp和的密文（可用于softmax分母）
 *   error                  - 错误信息
 *
 */

// func CiphertextMatricesComputeAttentionWithBSGSAndApproxMax(
// 	ctQ, ctK, ctV *he.CiphertextMatrices,
// 	LayerNumber int,
// 	modelParams *configs.ModelParams,
// 	ckksParams ckks.Parameters,
// 	eval *ckks.Evaluator,
// ) (*he.CiphertextMatrices, *rlwe.Ciphertext, error) {
// 	ctRows := ctQ.NumRow
// 	ctCols := ctQ.NumCol
// 	ctBatch := ctQ.NumBatch

// 	babyStep := modelParams.BabyStep
// 	giantStep := modelParams.GiantStep
// 	baseLen := modelParams.NumBatch
// 	realRow := float64(modelParams.NumRealRow)

// 	// 生成掩码矩阵
// 	maskMat := utils.GenerateSpecialSquareMatrix(modelParams)

// 	// 检查QKV是否一致
// 	if ctQ.NumBatch != ctK.NumBatch || ctQ.NumBatch != ctV.NumBatch {
// 		return nil, nil, fmt.Errorf("ctQ, ctK and ctV must have the same Batch: ctQ:%d , ctK:%d , ctV: %d", ctQ.NumBatch, ctK.NumBatch, ctV.NumBatch)
// 	}
// 	if ctQ.NumRow != ctK.NumRow || ctQ.NumRow != ctV.NumRow {
// 		return nil, nil, fmt.Errorf("ctQ, ctK and ctV must have the same Row: ctQ:%d , ctK:%d , ctV: %d", ctQ.NumRow, ctK.NumRow, ctV.NumRow)
// 	}
// 	if ctQ.NumCol != ctK.NumCol || ctQ.NumCol != ctV.NumCol {
// 		return nil, nil, fmt.Errorf("ctQ, ctK and ctV must have the same Col: ctQ:%d , ctK:%d , ctV: %d", ctQ.NumCol, ctK.NumCol, ctV.NumCol)
// 	}

// 	// fmt.Println(babyStep, giantStep)
// 	// 生成所有步长
// 	babySteps := make([]int, 0, babyStep)
// 	for i := 0; i < babyStep; i++ {
// 		babySteps = append(babySteps, i*baseLen)
// 	}
// 	giantSteps := make([]int, 0, giantStep)
// 	for i := 0; i < giantStep; i++ {
// 		giantSteps = append(giantSteps, -i*baseLen*babyStep)
// 	}

// 	// 使用 hoisting 技术批量旋转所有的密文矩阵
// 	ctQRotated := RotateCiphertextMatricesHoisting(ctQ, giantSteps, eval)
// 	ctKRotated := RotateCiphertextMatricesHoisting(ctK, babySteps, eval)
// 	ctVRotated := RotateCiphertextMatricesHoisting(ctV, babySteps, eval)

// 	ctQKTCiphertext := make([]*rlwe.Ciphertext, ctRows)
// 	ctQKTAdd := ckks.NewCiphertext(ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
// 	// Step1. Compute Add All Row
// 	for i := 0; i < giantStep; i++ {
// 		// 用于存储每个giantStep的结果
// 		ctAddGaintStep := ckks.NewCiphertext(ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
// 		for j := 0; j < babyStep; j++ {
// 			// 检查是否超出行数
// 			if i*babyStep+j >= ctRows {
// 				break // 超出行数，退出循环
// 			}

// 			// step1.1: 计算 ctQRotated[i] 和 ctKRotated[j] 的乘积
// 			ct, err := CiphertextMatricesMultiplyCiphertextMatricesThenAdd(ctQRotated[i], ctKRotated[j], &ckksParams, eval)
// 			if err != nil {
// 				return nil, nil, fmt.Errorf("failed to multiply ctQRotated[%d] and ctKRotated[%d]: %w", i, j, err)
// 			}

// 			// step1.2:用于求和SUM(ctQKT)
// 			err = eval.Add(ctAddGaintStep, ct, ctAddGaintStep)
// 			if err != nil {
// 				panic(err)
// 			}
// 			ctQKTCiphertext[i*babyStep+j] = ct
// 		}
// 		// 用于求和SUM(ctQKT)
// 		ctAddRotate, err := eval.RotateNew(ctAddGaintStep, i*baseLen*babyStep)
// 		if err != nil {
// 			panic(err)
// 		}
// 		err = eval.Add(ctQKTAdd, ctAddRotate, ctQKTAdd)
// 		if err != nil {
// 			panic(err)
// 		}
// 	}

// 	// Step2. Compute Approx Max
// 	ctMean, err := eval.MulRelinNew(ctQKTAdd, 1/float64(modelParams.NumRealRow))
// 	if err != nil {
// 		panic(err)
// 	}
// 	if err = eval.Rescale(ctMean, ctMean); err != nil {
// 		panic(err)
// 	}
// 	ctMeanRot, err := eval.RotateHoistedNew(ctMean, giantSteps)
// 	if err != nil {
// 		panic(err)
// 	}

// 	ctVarNotdivd := ckks.NewCiphertext(ckksParams, ctMean.Degree(), ctMean.Level())
// 	for i := 0; i < giantStep; i++ {
// 		// 用于存储每个giantStep的结果
// 		ctAddGaintStep := ckks.NewCiphertext(ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
// 		for j := 0; j < babyStep; j++ {
// 			// 检查是否超出行数
// 			if i*babyStep+j >= ctRows {
// 				break // 超出行数，退出循环
// 			}
// 			// Sub Mean Value
// 			ctSub, err := eval.SubNew(ctQKTCiphertext[i*babyStep+j], ctMeanRot[-i*baseLen*babyStep])
// 			if err != nil {
// 				panic(err)
// 			}

// 			// Sub Mean Valuen Square
// 			if err = eval.MulRelin(ctSub, ctSub, ctSub); err != nil {
// 				panic(err)
// 			}
// 			if err = eval.Rescale(ctSub, ctSub); err != nil {
// 				panic(err)
// 			}

// 			maskVec := utils.ExtractAndRepeatDiagonal(maskMat, i*babyStep+j, modelParams.NumBatch)
// 			maskVecRot := utils.RotateSliceNew(maskVec, -i*baseLen*babyStep)
// 			if err = eval.MulRelin(ctSub, maskVecRot, ctSub); err != nil {
// 				panic(err)
// 			}
// 			if err = eval.Rescale(ctSub, ctSub); err != nil {
// 				panic(err)
// 			}

// 			// accumulate add gaint value
// 			err = eval.Add(ctAddGaintStep, ctSub, ctAddGaintStep)
// 			if err != nil {
// 				panic(err)
// 			}
// 		}
// 		// 用于求和SUM(ctQKT)
// 		ctAddRotate, err := eval.RotateNew(ctAddGaintStep, i*baseLen*babyStep)
// 		if err != nil {
// 			panic(err)
// 		}
// 		err = eval.Add(ctVarNotdivd, ctAddRotate, ctVarNotdivd)
// 		if err != nil {
// 			panic(err)
// 		}
// 	}
// 	constantValue := 2.08
// 	varScale := 0.1 * (1.0 / realRow) * constantValue
// 	constValue := 1.25 * constantValue
// 	ctVarScale, err := eval.MulRelinNew(ctVarNotdivd, varScale)
// 	fmt.Println(varScale, "  ", constValue)
// 	if err != nil {
// 		panic(err)
// 	}
// 	if err = eval.Rescale(ctVarScale, ctVarScale); err != nil {
// 		panic(err)
// 	}
// 	// approx std
// 	ctApproxStd, err := eval.AddNew(ctVarScale, constValue)
// 	if err != nil {
// 		panic(err)
// 	}
// 	ctApproxMax, err := eval.AddNew(ctApproxStd, ctMean)
// 	if err != nil {
// 		panic(err)
// 	}
// 	ctApproxMaxRot, err := eval.RotateHoistedNew(ctApproxMax, giantSteps)
// 	if err != nil {
// 		panic(err)
// 	}

// 	// Step3. Sub approx max and compute QK
// 	newCiphertexts := make([]*rlwe.Ciphertext, ctCols)
// 	for k := 0; k < ctCols; k++ {
// 		newCiphertexts[k] = ckks.NewCiphertext(ckksParams, ctApproxMax.Degree(), ctApproxMax.Level())
// 	}
// 	ctAddAllbyRow := ckks.NewCiphertext(ckksParams, ctApproxMax.Degree(), ctApproxMax.Level())
// 	for i := 0; i < giantStep; i++ {

// 		// 用于存储每个giantStep的结果
// 		ctAddGaintStep := ckks.NewCiphertext(ckksParams, ctApproxMax.Degree(), ctApproxMax.Level())
// 		// 这里的ctQKV是一个新的CiphertextMatrices，用于存储 QKV 的结果
// 		localNewCiphertexts := make([]*rlwe.Ciphertext, ctV.NumCol)
// 		for k := 0; k < ctV.NumCol; k++ {
// 			localNewCiphertexts[k] = ckks.NewCiphertext(ckksParams, ctApproxMax.Degree(), ctApproxMax.Level())
// 		}
// 		ctQKV := &he.CiphertextMatrices{
// 			Ciphertexts: localNewCiphertexts,
// 			NumBatch:    ctBatch,
// 			NumRow:      ctRows,
// 			NumCol:      ctCols,
// 		}
// 		for j := 0; j < babyStep; j++ {
// 			// 检查是否超出行数
// 			if i*babyStep+j >= ctRows {
// 				break // 超出行数，退出循环
// 			}

// 			// step1: 计算 减去最大值
// 			ct, err := eval.SubNew(ctQKTCiphertext[i*babyStep+j], ctApproxMaxRot[-i*baseLen*babyStep])
// 			if err != nil {
// 				panic(err)
// 			}
// 			// step2: 计算exp函数
// 			ctExp := softmax.CiphertextExpChebyshev(ct, &ckksParams, eval)

// 			// step3：计算乘以mask attention
// 			maskVec := utils.ExtractAndRepeatDiagonal(maskMat, i*babyStep+j, modelParams.NumBatch)
// 			maskVecRot := utils.RotateSliceNew(maskVec, -i*baseLen*babyStep)
// 			if err = eval.MulRelin(ctExp, maskVecRot, ctExp); err != nil {
// 				panic(err)
// 			}
// 			if err = eval.Rescale(ctExp, ctExp); err != nil {
// 				panic(err)
// 			}

// 			// step4: 将Exp(ct - ctApproxMax) * ctVRotated[j] 存入ctQKV
// 			err = CiphertextMatricesMultiplyCiphertextAddToRes(ctVRotated[j], ctExp, ctQKV, &ckksParams, eval)
// 			if err != nil {
// 				panic(err)

// 			}

// 			// 用于求和SUM(Exp(ct - modelParams.ExpSubValue))
// 			err = eval.Add(ctAddGaintStep, ctExp, ctAddGaintStep)
// 			if err != nil {
// 				panic(err)
// 			}
// 		}
// 		// 旋转QKV的结果
// 		QKVRotKi := RotateCiphertextMatrices(ctQKV, modelParams, i*baseLen*babyStep, eval)
// 		// 将QKVRotKi的结果累加到newCiphertexts中
// 		for k := 0; k < ctCols; k++ {
// 			err := eval.Add(newCiphertexts[k], QKVRotKi.Ciphertexts[k], newCiphertexts[k])
// 			if err != nil {
// 				panic(err)
// 			}
// 		}

// 		// 用于求和SUM(Exp(ct - modelParams.ExpSubValue))
// 		ctAddRotate, err := eval.RotateNew(ctAddGaintStep, i*baseLen*babyStep)
// 		if err != nil {
// 			panic(err)
// 		}
// 		err = eval.Add(ctAddAllbyRow, ctAddRotate, ctAddAllbyRow)
// 		if err != nil {
// 			panic(err)
// 		}
// 	}

// 	return &he.CiphertextMatrices{
// 		Ciphertexts: newCiphertexts,
// 		NumBatch:    ctBatch,
// 		NumRow:      ctRows,
// 		NumCol:      ctCols,
// 	}, ctAddAllbyRow, nil
// }

/*
 * CiphertextMatricesComputeAttentionWithBSGSAndApproxMaxNotMask
 * 功能：
 *   对三个密文矩阵 ctQ、ctK、ctV，先用 Halevi-Shoup（大步-小步+hoisting）策略计算 Q × K^T，
 *   然后对每个乘积结果减去 modelParams.ExpSubValue，并对每个元素做指数函数（exp）近似（如Chebyshev多项式），
 *   再与V做乘法，最后对每一行的exp结果进行求和，返回所有exp结果的密文矩阵和每行和的密文。
 *   在计算时，最大值不用固定值，而使用近似值进行计算
 *   均值：rowMean  方差：rowVar  标准差：rowStd = 1.25 + 0.1 * rowVar
 *   近似最大值：rowMean + rowStd * sqrt(2 * log(RealRow))
 *
 * 输入参数：
 *   ctQ, ctK, ctV - 输入的密文矩阵（*he.CiphertextMatrices），维度需一致
 *   modelParams   - 模型参数（*configs.ModelParams），包含大步/小步参数和exp偏移
 *   ckksParams    - CKKS参数
 *   eval          - CKKS Evaluator
 *
 * 输出参数：
 *   *he.CiphertextMatrices - 经过exp处理并与V相乘后的密文矩阵（即attention输出）
 *   *rlwe.Ciphertext       - 每行exp和的密文（可用于softmax分母）
 *   error                  - 错误信息
 *
 */
func CiphertextMatricesComputeAttentionWithBSGSAndApproxMaxNotMask(
	ctQ, ctK, ctV *he.CiphertextMatrices,
	LayerNumber int,
	modelParams *configs.ModelParams,
	ckksParams ckks.Parameters,
	eval *ckks.Evaluator,
) (*he.CiphertextMatrices, *rlwe.Ciphertext, error) {
	ctRows := ctQ.NumRow
	ctCols := ctQ.NumCol
	ctBatch := ctQ.NumBatch

	babyStep := modelParams.BabyStep
	giantStep := modelParams.GiantStep
	baseLen := modelParams.NumBatch
	realRow := float64(modelParams.NumRealRow)

	// 检查QKV是否一致
	if ctQ.NumBatch != ctK.NumBatch || ctQ.NumBatch != ctV.NumBatch {
		return nil, nil, fmt.Errorf("ctQ, ctK and ctV must have the same Batch: ctQ:%d , ctK:%d , ctV: %d", ctQ.NumBatch, ctK.NumBatch, ctV.NumBatch)
	}
	if ctQ.NumRow != ctK.NumRow || ctQ.NumRow != ctV.NumRow {
		return nil, nil, fmt.Errorf("ctQ, ctK and ctV must have the same Row: ctQ:%d , ctK:%d , ctV: %d", ctQ.NumRow, ctK.NumRow, ctV.NumRow)
	}
	if ctQ.NumCol != ctK.NumCol || ctQ.NumCol != ctV.NumCol {
		return nil, nil, fmt.Errorf("ctQ, ctK and ctV must have the same Col: ctQ:%d , ctK:%d , ctV: %d", ctQ.NumCol, ctK.NumCol, ctV.NumCol)
	}

	// 生成所有步长
	babySteps := make([]int, 0, babyStep)
	for i := 0; i < babyStep; i++ {
		babySteps = append(babySteps, i*baseLen)
	}
	giantSteps := make([]int, 0, giantStep)
	for i := 0; i < giantStep; i++ {
		giantSteps = append(giantSteps, -i*baseLen*babyStep)
	}

	// 使用 hoisting 技术批量旋转所有的密文矩阵
	ctQRotated := RotateCiphertextMatricesHoisting(ctQ, giantSteps, eval)
	ctKRotated := RotateCiphertextMatricesHoisting(ctK, babySteps, eval)
	ctVRotated := RotateCiphertextMatricesHoisting(ctV, babySteps, eval)

	ctQKTCiphertext := make([]*rlwe.Ciphertext, ctRows)
	ctQKTAdd := ckks.NewCiphertext(ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
	// Step1. Compute Add All Row
	for i := 0; i < giantStep; i++ {
		// 用于存储每个giantStep的结果
		ctAddGaintStep := ckks.NewCiphertext(ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
		for j := 0; j < babyStep; j++ {
			// 检查是否超出行数
			if i*babyStep+j >= ctRows {
				break // 超出行数，退出循环
			}

			// step1.1: 计算 ctQRotated[i] 和 ctKRotated[j] 的乘积
			ct, err := CiphertextMatricesMultiplyCiphertextMatricesThenAdd(ctQRotated[i], ctKRotated[j], &ckksParams, eval)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to multiply ctQRotated[%d] and ctKRotated[%d]: %w", i, j, err)
			}

			// step1.2:用于求和SUM(ctQKT)
			err = eval.Add(ctAddGaintStep, ct, ctAddGaintStep)
			if err != nil {
				panic(err)
			}
			ctQKTCiphertext[i*babyStep+j] = ct
		}
		// 用于求和SUM(ctQKT)
		ctAddRotate, err := eval.RotateNew(ctAddGaintStep, i*baseLen*babyStep)
		if err != nil {
			panic(err)
		}
		err = eval.Add(ctQKTAdd, ctAddRotate, ctQKTAdd)
		if err != nil {
			panic(err)
		}
	}

	// Step2. Compute Approx Max
	ctMean, err := eval.MulRelinNew(ctQKTAdd, 1/float64(modelParams.NumRealRow))
	if err != nil {
		panic(err)
	}
	if err = eval.Rescale(ctMean, ctMean); err != nil {
		panic(err)
	}
	ctMeanRot, err := eval.RotateHoistedNew(ctMean, giantSteps)
	if err != nil {
		panic(err)
	}

	ctVarNotdivd := ckks.NewCiphertext(ckksParams, ctMean.Degree(), ctMean.Level())
	for i := 0; i < giantStep; i++ {
		// 用于存储每个giantStep的结果
		ctAddGaintStep := ckks.NewCiphertext(ckksParams, ctQ.Ciphertexts[0].Degree(), ctQ.Ciphertexts[0].Level())
		for j := 0; j < babyStep; j++ {
			// 检查是否超出行数
			if i*babyStep+j >= ctRows {
				break // 超出行数，退出循环
			}
			// Sub Mean Value
			ctSub, err := eval.SubNew(ctQKTCiphertext[i*babyStep+j], ctMeanRot[-i*baseLen*babyStep])
			if err != nil {
				panic(err)
			}

			// Sub Mean Valuen Square
			if err = eval.MulRelin(ctSub, ctSub, ctSub); err != nil {
				panic(err)
			}
			if err = eval.Rescale(ctSub, ctSub); err != nil {
				panic(err)
			}

			// accumulate add gaint value
			err = eval.Add(ctAddGaintStep, ctSub, ctAddGaintStep)
			if err != nil {
				panic(err)
			}
		}
		// 用于求和SUM(ctQKT)
		ctAddRotate, err := eval.RotateNew(ctAddGaintStep, i*baseLen*babyStep)
		if err != nil {
			panic(err)
		}
		err = eval.Add(ctVarNotdivd, ctAddRotate, ctVarNotdivd)
		if err != nil {
			panic(err)
		}
	}
	constantValue := 2.08
	varScale := 0.1 * (1.0 / realRow) * constantValue
	constValue := 1.25 * constantValue
	ctVarScale, err := eval.MulRelinNew(ctVarNotdivd, varScale)
	// fmt.Println(varScale, "  ", constValue)
	if err != nil {
		panic(err)
	}
	if err = eval.Rescale(ctVarScale, ctVarScale); err != nil {
		panic(err)
	}
	// approx std
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

	// Step3. Sub approx max and compute QK
	newCiphertexts := make([]*rlwe.Ciphertext, ctCols)
	for k := 0; k < ctCols; k++ {
		newCiphertexts[k] = ckks.NewCiphertext(ckksParams, ctApproxMax.Degree(), ctApproxMax.Level())
	}
	ctAddAllbyRow := ckks.NewCiphertext(ckksParams, ctApproxMax.Degree(), ctApproxMax.Level())
	for i := 0; i < giantStep; i++ {

		// 用于存储每个giantStep的结果
		ctAddGaintStep := ckks.NewCiphertext(ckksParams, ctApproxMax.Degree(), ctApproxMax.Level())
		// 这里的ctQKV是一个新的CiphertextMatrices，用于存储 QKV 的结果
		localNewCiphertexts := make([]*rlwe.Ciphertext, ctV.NumCol)
		for k := 0; k < ctV.NumCol; k++ {
			localNewCiphertexts[k] = ckks.NewCiphertext(ckksParams, ctApproxMax.Degree(), ctApproxMax.Level())
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

			// step1: 计算 减去最大值
			ct, err := eval.SubNew(ctQKTCiphertext[i*babyStep+j], ctApproxMaxRot[-i*baseLen*babyStep])
			if err != nil {
				panic(err)
			}
			// step2: 计算exp函数
			ctExp := softmax.CiphertextExpChebyshev(ct, &ckksParams, eval, modelParams.ExpMinValue, modelParams.ExpMaxValue, modelParams.ExpDegree)

			// step4: 将Exp(ct - ctApproxMax) * ctVRotated[j] 存入ctQKV
			err = CiphertextMatricesMultiplyCiphertextAddToRes(ctVRotated[j], ctExp, ctQKV, &ckksParams, eval)
			if err != nil {
				panic(err)

			}

			// 用于求和SUM(Exp(ct - modelParams.ExpSubValue))
			err = eval.Add(ctAddGaintStep, ctExp, ctAddGaintStep)
			if err != nil {
				panic(err)
			}
		}
		// 旋转QKV的结果
		QKVRotKi := RotateCiphertextMatrices(ctQKV, i*baseLen*babyStep, eval)
		// 将QKVRotKi的结果累加到newCiphertexts中
		for k := 0; k < ctCols; k++ {
			err := eval.Add(newCiphertexts[k], QKVRotKi.Ciphertexts[k], newCiphertexts[k])
			if err != nil {
				panic(err)
			}
		}

		// 用于求和SUM(Exp(ct - modelParams.ExpSubValue))
		ctAddRotate, err := eval.RotateNew(ctAddGaintStep, i*baseLen*babyStep)
		if err != nil {
			panic(err)
		}
		err = eval.Add(ctAddAllbyRow, ctAddRotate, ctAddAllbyRow)
		if err != nil {
			panic(err)
		}
	}

	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ctBatch,
		NumRow:      ctRows,
		NumCol:      ctCols,
	}, ctAddAllbyRow, nil
}

/*
 * CiphertextMatricesMultiplyCiphertextMatricesThenAdd
 * Input:  Q,K he.CiphertextMatrices,ckks.Parameters,ckks.Evaluator
 * Output: *rlwe.Ciphertext,error
 * Compute: Add(Mul(Q_i, K_i)) for i=0 to NumCol-1
 */
func CiphertextMatricesMultiplyCiphertextMatricesThenAdd(
	cipherMats1 *he.CiphertextMatrices,
	cipherMats2 *he.CiphertextMatrices,
	ckksParam *ckks.Parameters,
	eval *ckks.Evaluator,
) (*rlwe.Ciphertext, error) {

	// 判断QKV是否一样
	if cipherMats1.NumCol != cipherMats2.NumCol {
		return nil, fmt.Errorf("cipherMats1.NumRow(%d) != cipherMats2.NumRow(%d)", cipherMats1.NumRow, cipherMats2.NumRow)
	}

	// ct := rlwe.NewCiphertext(ckksParam, cipherMats1.Ciphertexts[0].Degree(), cipherMats1.Ciphertexts[0].Level())
	ct, _ := eval.MulRelinNew(cipherMats1.Ciphertexts[0], cipherMats2.Ciphertexts[0])
	// 进行BSGS To Attetion
	for i := 1; i < cipherMats1.NumCol; i++ {
		err := eval.MulRelinThenAdd(cipherMats1.Ciphertexts[i], cipherMats2.Ciphertexts[i], ct)
		if err != nil {
			panic(err)
		}
	}
	eval.Rescale(ct, ct)

	ct.Scale = ckksParam.DefaultScale()
	// ctScale := &ct.Scale.Value // We need to access the pointer in order for it to display correctly in the command line.
	// fmt.Printf("CiphertextMatricesMultiplyCiphertextMatricesThenAdd Scale rescaling: %f\n", ctScale)
	// 返回结果
	return ct, nil
}

/*
 * CiphertextMatricesMultiplyCiphertextAddToRes
 * Input:  CiphertextMatrices,rlwe.Ciphertext,he.CiphertextMatrices,ckks.Parameters,ckks.Evaluator
 * Output: error(destination is modified)
 * Compute: Q,K,V --> Attention result
 */
func CiphertextMatricesMultiplyCiphertextAddToRes(
	cipherMats *he.CiphertextMatrices, // 按列存放的密文矩阵
	ct1 *rlwe.Ciphertext, // 与每列相乘的权重/标量密文
	destination *he.CiphertextMatrices, // 结果累加到这里（逐列）
	ckksParam *ckks.Parameters,
	eval *ckks.Evaluator,
) error {
	// 基本校验
	if cipherMats == nil || ct1 == nil || destination == nil {
		return fmt.Errorf("nil input: cipherMats=%v ct1=%v destination=%v",
			cipherMats == nil, ct1 == nil, destination == nil)
	}
	if cipherMats.NumCol <= 0 || len(cipherMats.Ciphertexts) != cipherMats.NumCol {
		return fmt.Errorf("cipherMats shape invalid: NumCol=%d len=%d",
			cipherMats.NumCol, len(cipherMats.Ciphertexts))
	}
	// 目的矩阵列数不对就重建（只分配 slice，不要预填零密文）
	if destination.Ciphertexts == nil || len(destination.Ciphertexts) != cipherMats.NumCol {
		destination.Ciphertexts = make([]*rlwe.Ciphertext, cipherMats.NumCol)
		destination.NumCol = cipherMats.NumCol
		// 保持元信息一致（按需）
		destination.NumRow = cipherMats.NumRow
		destination.NumBatch = cipherMats.NumBatch
	}

	// 逐列：prod = cipherMats[i] * ct1；destination[i] += prod
	for i := 0; i < cipherMats.NumCol; i++ {
		a := cipherMats.Ciphertexts[i]
		if a == nil {
			continue // 允许稀疏
		}
		prod, err := eval.MulRelinNew(a, ct1)
		if err != nil {
			return fmt.Errorf("MulRelinNew col=%d: %w", i, err)
		}
		eval.Rescale(prod, prod)

		// ★ 首项用 CopyNew 初始化累加器，避免 nil 传给 Add 以及 scale/level 不对齐
		if destination.Ciphertexts[i] == nil {
			destination.Ciphertexts[i] = prod.CopyNew()
		} else {
			// 这里假定 level/scale 已在上游一致；如有需要，可在此处做 DropLevel 对齐
			eval.Add(destination.Ciphertexts[i], prod, destination.Ciphertexts[i])
		}
	}
	return nil
}

// CiphertextMatricesAddCiphertextMatrices
// 输入:  ctMats1, ctMats2 *he.CiphertextMatrices
// 输出: *he.CiphertextMatrices, error
// 说明: 对两个 CiphertextMatrices 中每个对应密文做加法，返回新的 CiphertextMatrices
func CiphertextMatricesAddCiphertextMatrices(
	ctMats1 *he.CiphertextMatrices,
	ctMats2 *he.CiphertextMatrices,
	eval *ckks.Evaluator,
) (*he.CiphertextMatrices, error) {
	if ctMats1.NumBatch != ctMats2.NumBatch ||
		ctMats1.NumRow != ctMats2.NumRow ||
		ctMats1.NumCol != ctMats2.NumCol ||
		len(ctMats1.Ciphertexts) != len(ctMats2.Ciphertexts) {
		return nil, fmt.Errorf("CiphertextMatricesAddCiphertextMatrices: 维度不匹配")
	}

	newCiphertexts := make([]*rlwe.Ciphertext, len(ctMats1.Ciphertexts))
	for i := 0; i < len(ctMats1.Ciphertexts); i++ {
		ct, err := eval.AddNew(ctMats1.Ciphertexts[i], ctMats2.Ciphertexts[i])
		if err != nil {
			return nil, fmt.Errorf("CiphertextMatricesAddCiphertextMatrices: 第%d个密文加法失败: %v", i, err)
		}
		newCiphertexts[i] = ct
	}

	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ctMats1.NumBatch,
		NumRow:      ctMats1.NumRow,
		NumCol:      ctMats1.NumCol,
	}, nil
}

// CiphertextMatricesMulCiphertextMatrices
// 输入:  ctMats1, ctMats2 *he.CiphertextMatrices
// 输出: *he.CiphertextMatrices, error
// 说明: 对两个 CiphertextMatrices 中每个对应密文做「点对点乘法」，返回新的 CiphertextMatrices
func CiphertextMatricesMulCiphertextMatrices(
	ctMats1 *he.CiphertextMatrices,
	ctMats2 *he.CiphertextMatrices,
	eval *ckks.Evaluator,
) (*he.CiphertextMatrices, error) {

	// 1. 维度检查
	if ctMats1.NumBatch != ctMats2.NumBatch ||
		ctMats1.NumRow != ctMats2.NumRow ||
		ctMats1.NumCol != ctMats2.NumCol ||
		len(ctMats1.Ciphertexts) != len(ctMats2.Ciphertexts) {
		return nil, fmt.Errorf("CiphertextMatricesMulCiphertextMatrices: 维度不匹配")
	}

	// 2. 逐个密文相乘
	newCiphertexts := make([]*rlwe.Ciphertext, len(ctMats1.Ciphertexts))
	for i := 0; i < len(ctMats1.Ciphertexts); i++ {
		ct1 := ctMats1.Ciphertexts[i]
		ct2 := ctMats2.Ciphertexts[i]

		ct, err := eval.MulRelinNew(ct1, ct2)
		if err != nil {
			return nil, fmt.Errorf("CiphertextMatricesMulCiphertextMatrices: 第 %d 个密文乘法失败: %v", i, err)
		}

		if err := eval.Rescale(ct, ct); err != nil {
			panic(err)
		}

		newCiphertexts[i] = ct
	}

	// 3. 返回新的 CiphertextMatrices
	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ctMats1.NumBatch,
		NumRow:      ctMats1.NumRow,
		NumCol:      ctMats1.NumCol,
	}, nil
}
