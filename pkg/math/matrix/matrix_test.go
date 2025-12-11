package matrix

import (
	"Arion/configs"
	"Arion/pkg/ecdmat"
	"Arion/pkg/he"
	"Arion/pkg/utils"
	"fmt"
	"testing"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"gonum.org/v1/gonum/mat"
)

func TestCiphertextMatricesMultiplyPlaintextMatrix(t *testing.T) {
	// 使用小规模模型参数
	modelParams := &configs.ModelParams{
		NumBatch:   4,
		NumRow:     8,
		NumCol:     768,
		SqrtD:      2.0,
		NumRealRow: 5,
	}

	// 初始化简单的CKKS参数
	ckksParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            6,
		LogQ:            []int{58, 45, 45, 45},
		LogP:            []int{60},
		LogDefaultScale: 45,
	})
	if err != nil {
		t.Fatalf("CKKS参数初始化失败: %v", err)
	}

	// 构造有规律的输入矩阵 input
	input := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
	for i := 0; i < modelParams.NumRow; i++ {
		for j := 0; j < modelParams.NumCol; j++ {
			input.Set(i, j, float64(i*modelParams.NumCol+j+1)/10) // 1,2,3,4,5...
		}
	}
	// 编码、批处理
	input = utils.PadOrTruncateMatrix(input, modelParams)
	inputMats := utils.MatrixToBatchMats(input, modelParams)
	encodedInputs := ecdmat.EncodeDense(inputMats, modelParams)

	// 生成密钥
	ecd, enc, eval, dec := he.GenerateKeys(ckksParams, modelParams)

	// 加密
	ctX, err := he.EncryptInputMatrices(encodedInputs, modelParams, ckksParams, enc, ecd)
	if err != nil {
		t.Fatalf("加密失败: %v", err)
	}

	// 构造有规律的明文权重矩阵（NumCol × NumCol）
	weight := mat.NewDense(modelParams.NumCol, modelParams.NumCol, nil)
	for i := 0; i < modelParams.NumCol; i++ {
		for j := 0; j < modelParams.NumCol; j++ {
			weight.Set(i, j, float64(i+j+1)/100) // 1,2,3,4,5...
		}
	}

	// 密文矩阵 × 明文矩阵
	ctY, err := CiphertextMatricesMultiplyPlaintextMatrix(ctX, modelParams, weight, ckksParams, eval)
	if err != nil {
		t.Fatalf("密文矩阵与明文矩阵乘法失败: %v", err)
	}

	// 解密
	decY, err := he.DecryptCiphertextMatrices(ctY, modelParams, ckksParams, dec, ecd)
	if err != nil {
		t.Fatalf("解密失败: %v", err)
	}

	for i := 0; i < len(decY); i++ {
		fmt.Printf("解密后的矩阵 %d: %v\n", i, decY[i])
	}

	// 解码
	matsY := ecdmat.DecodeDense(decY, modelParams)

	// 输出输入矩阵 input
	t.Log("输入矩阵（input）：")
	utils.PrintMat(input)

	// 输出权重矩阵 weight
	t.Log("权重矩阵（weight）：")
	utils.PrintMat(weight)

	// 明文直接计算结果
	expected := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
	expected.Product(input, weight)

	// 输出明文计算结果
	t.Log("明文计算结果（expected）：")
	utils.PrintMat(expected)

	// 检查第一个batch的结果与明文计算是否接近
	got := matsY[0]
	rows, cols := got.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			want := expected.At(i, j)
			val := got.At(i, j)
			if diff := abs(want - val); diff > 1e-2 {
				t.Errorf("结果不符: got %.4f, want %.4f, diff=%.4f (i=%d, j=%d)", val, want, diff, i, j)
			}
		}
	}
	utils.PrintMat(got)
	utils.PrintMat(matsY[1]) // 打印第二个batch的结果，验证是否正确
}

func TestCiphertextMatricesMultiplyWeightAndAddBias(t *testing.T) {
	// 使用小规模模型参数
	modelParams := &configs.ModelParams{
		NumBatch:   4,
		NumRow:     2,
		NumCol:     4,
		SqrtD:      2.0,
		NumRealRow: 2,
	}

	// 初始化简单的CKKS参数
	ckksParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            4,
		LogQ:            []int{60, 40, 40, 60},
		LogP:            []int{60},
		LogDefaultScale: 40,
	})
	if err != nil {
		t.Fatalf("CKKS参数初始化失败: %v", err)
	}

	// 构造有规律的输入矩阵 input
	input := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
	for i := 0; i < modelParams.NumRow; i++ {
		for j := 0; j < modelParams.NumCol; j++ {
			input.Set(i, j, float64(i*modelParams.NumCol+j+1)/10) // 1,2,3,...
		}
	}
	// 编码、批处理
	input = utils.PadOrTruncateMatrix(input, modelParams)
	inputMats := utils.MatrixToBatchMats(input, modelParams)
	encodedInputs := ecdmat.EncodeDense(inputMats, modelParams)

	// 生成密钥
	ecd, enc, eval, dec := he.GenerateKeys(ckksParams, modelParams)

	// 加密
	ctX, err := he.EncryptInputMatrices(encodedInputs, modelParams, ckksParams, enc, ecd)
	if err != nil {
		t.Fatalf("加密失败: %v", err)
	}

	// 构造有规律的明文权重矩阵（NumCol × d），这里d=2
	d := 2
	weight := mat.NewDense(modelParams.NumCol, d, nil)
	for i := 0; i < modelParams.NumCol; i++ {
		for j := 0; j < d; j++ {
			weight.Set(i, j, float64(i+j+1)/10) // 1,2,3,...
		}
	}

	// 构造有规律的明文偏置向量（d×1）
	bias := mat.NewDense(d, 1, nil)
	for i := 0; i < d; i++ {
		bias.Set(i, 0, float64(i+1)/10) // 1,2,...
	}

	// 密文矩阵 × 明文权重矩阵 + 明文偏置
	ctY, err := CiphertextMatricesMultiplyWeightAndAddBias(ctX, weight, bias, modelParams, ckksParams, eval)
	if err != nil {
		t.Fatalf("密文矩阵与明文权重矩阵乘法加偏置失败: %v", err)
	}

	// 解密
	decY, err := he.DecryptCiphertextMatrices(ctY, &configs.ModelParams{
		NumBatch: modelParams.NumBatch,
		NumRow:   modelParams.NumRow,
		NumCol:   d,
	}, ckksParams, dec, ecd)
	if err != nil {
		t.Fatalf("解密失败: %v", err)
	}

	// 解码
	matsY := ecdmat.DecodeDense(decY, &configs.ModelParams{
		NumBatch: modelParams.NumBatch,
		NumRow:   modelParams.NumRow,
		NumCol:   d,
	})

	// 输出输入矩阵 input
	t.Log("输入矩阵（input）：")
	utils.PrintMat(input)

	// 输出权重矩阵 weight
	t.Log("权重矩阵（weight）：")
	utils.PrintMat(weight)

	// 输出偏置向量 bias
	t.Log("偏置向量（bias）：")
	utils.PrintMat(bias)

	// 明文直接计算结果 expected = input * weight + bias
	expected := mat.NewDense(modelParams.NumRow, d, nil)
	expected.Product(input, weight)
	for i := 0; i < modelParams.NumRow; i++ {
		for j := 0; j < d; j++ {
			expected.Set(i, j, expected.At(i, j)+bias.At(j, 0))
		}
	}

	// 输出明文计算结果
	t.Log("明文计算结果（expected）：")
	utils.PrintMat(expected)

	// 检查第一个batch的结果与明文计算是否接近
	got := matsY[0]
	rows, cols := got.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			want := expected.At(i, j)
			val := got.At(i, j)
			if diff := abs(want - val); diff > 1e-2 {
				t.Errorf("结果不符: got %.4f, want %.4f, diff=%.4f (i=%d, j=%d)", val, want, diff, i, j)
			}
		}
	}
	utils.PrintMat(got)
	if len(matsY) > 1 {
		utils.PrintMat(matsY[1]) // 打印第二个batch的结果，验证是否正确
	}
}

// func TestRotateCiphertextMatricesHoisting(t *testing.T) {
// 	// 使用小规模模型参数
// 	modelParams := &configs.ModelParams{
// 		NumBatch: 2,
// 		NumRow:   8,
// 		NumCol:   4,
// 		SqrtD:    2.0,
// 	}

// 	// 初始化简单的CKKS参数
// 	ckksParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
// 		LogN:            5,
// 		LogQ:            []int{60, 40, 40, 60},
// 		LogP:            []int{60},
// 		LogDefaultScale: 40,
// 	})
// 	if err != nil {
// 		t.Fatalf("CKKS参数初始化失败: %v", err)
// 	}

// 	// 构造有规律的输入矩阵 input
// 	input := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
// 	for i := 0; i < modelParams.NumRow; i++ {
// 		for j := 0; j < modelParams.NumCol; j++ {
// 			input.Set(i, j, float64(i*modelParams.NumCol+j+1))
// 		}
// 	}
// 	// 编码、批处理
// 	input = utils.PadOrTruncateMatrix(input, modelParams)
// 	inputMats := utils.MatrixToBatchMats(input, modelParams)
// 	encodedInputs := ecdmat.EncodeDense(inputMats, modelParams)

// 	// 生成密钥
// 	ecd, enc, eval, dec := he.GenerateKeys(ckksParams, modelParams)

// 	// 加密
// 	ctX, err := he.EncryptInputMatrices(encodedInputs, modelParams, ckksParams, enc, ecd)
// 	if err != nil {
// 		t.Fatalf("加密失败: %v", err)
// 	}

// 	// 测试旋转步长
// 	stepsSlice := []int{2, 4, 6}

// 	// 调用批量旋转函数
// 	rotatedList := RotateCiphertextMatricesHoisting(ctX, stepsSlice, eval)

// 	// 解密并输出旋转后的结果
// 	for idx, rotated := range rotatedList {
// 		decY, err := he.DecryptCiphertextMatrices(rotated, modelParams, ckksParams, dec, ecd)
// 		if err != nil {
// 			t.Fatalf("解密失败: %v", err)
// 		}
// 		matsY := ecdmat.DecodeDense(decY, modelParams)
// 		t.Logf("旋转步长 %d 后第一个batch的明文矩阵：", stepsSlice[idx])
// 		utils.PrintMat(matsY[0])
// 	}
// }

// func TestCiphertextMatricesMultiplyCiphertextMatricesToHaleviShoup(t *testing.T) {
// 	// 使用小规模模型参数
// 	modelParams := &configs.ModelParams{
// 		NumBatch:  2,
// 		NumRow:    8,
// 		NumCol:    4,
// 		SqrtD:     2.0,
// 		BabyStep:  3,
// 		GiantStep: 3,
// 	}

// 	// 初始化简单的CKKS参数
// 	ckksParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
// 		LogN:            5,
// 		LogQ:            []int{60, 40, 40, 60},
// 		LogP:            []int{60},
// 		LogDefaultScale: 40,
// 	})
// 	if err != nil {
// 		t.Fatalf("CKKS参数初始化失败: %v", err)
// 	}

// 	// 构造有规律的输入矩阵 inputQ
// 	inputQ := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
// 	for i := 0; i < modelParams.NumRow; i++ {
// 		for j := 0; j < modelParams.NumCol; j++ {
// 			inputQ.Set(i, j, float64(i*modelParams.NumCol+j+1)) // 1,2,3,4,...
// 		}
// 	}
// 	// 构造有规律但不同的输入矩阵 inputK，避免对称
// 	inputK := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
// 	for i := 0; i < modelParams.NumRow; i++ {
// 		for j := 0; j < modelParams.NumCol; j++ {
// 			inputK.Set(i, j, float64((i+1)*(j+2))) // 2,4,3,6,4,8,... 非对称
// 		}
// 	}

// 	// 编码、批处理
// 	inputQ = utils.PadOrTruncateMatrix(inputQ, modelParams)
// 	inputK = utils.PadOrTruncateMatrix(inputK, modelParams)
// 	inputMatsQ := utils.MatrixToBatchMats(inputQ, modelParams)
// 	inputMatsK := utils.MatrixToBatchMats(inputK, modelParams)
// 	encodedInputsQ := ecdmat.EncodeDense(inputMatsQ, modelParams)
// 	encodedInputsK := ecdmat.EncodeDense(inputMatsK, modelParams)

// 	// 生成密钥
// 	ecd, enc, eval, dec := he.GenerateKeys(ckksParams, modelParams)

// 	// 加密
// 	ctQ, err := he.EncryptInputMatrices(encodedInputsQ, modelParams, ckksParams, enc, ecd)
// 	if err != nil {
// 		t.Fatalf("加密Q失败: %v", err)
// 	}
// 	ctK, err := he.EncryptInputMatrices(encodedInputsK, modelParams, ckksParams, enc, ecd)
// 	if err != nil {
// 		t.Fatalf("加密K失败: %v", err)
// 	}

// 	// 调用Halevi-Shoup密文矩阵乘法
// 	ctQK, err := CiphertextMatricesMultiplyCiphertextMatricesToHaleviShoup(ctQ, ctK, modelParams, ckksParams, eval)
// 	if err != nil {
// 		t.Fatalf("Halevi-Shoup密文矩阵乘法失败: %v", err)
// 	}

// 	// 解密
// 	decY, err := he.DecryptCiphertextMatrices(ctQK, modelParams, ckksParams, dec, ecd)
// 	if err != nil {
// 		t.Fatalf("解密失败: %v", err)
// 	}

// 	// 解码
// 	matsY := ecdmat.DecodeDense(decY, modelParams)

// 	// 输出输入矩阵 inputQ
// 	t.Log("输入矩阵Q（inputQ）：")
// 	utils.PrintMat(inputQ)
// 	// 输出输入矩阵 inputK
// 	t.Log("输入矩阵K（inputK）：")
// 	utils.PrintMat(inputK)

// 	// 明文直接计算 Q × K^T
// 	expected := mat.NewDense(modelParams.NumRow, modelParams.NumRow, nil)
// 	kT := mat.DenseCopyOf(inputK.T())
// 	expected.Product(inputQ, kT)

// 	// 输出明文计算结果
// 	t.Log("明文 QK^T 结果（expected）：")
// 	utils.PrintMat(expected)

// 	// 检查第一个batch的结果与明文计算是否接近
// 	got := matsY[0]
// 	t.Log("密文计算 QK^T BSGS对角线编码结果：")
// 	utils.PrintMat(got)
// }

// func TestCiphertextMatricesComputeAttentionWithBSGS(t *testing.T) {
// 	// 使用小规模模型参数
// 	modelParams := &configs.ModelParams{
// 		NumBatch:    2,
// 		NumRow:      8,
// 		NumCol:      4,
// 		SqrtD:       2.0,
// 		BabyStep:    3,
// 		GiantStep:   3,
// 		ExpSubValue: []float64{0.0},
// 		NumRealRow:  5,
// 	}

// 	// 初始化简单的CKKS参数
// 	ckksParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
// 		LogN:            5,
// 		LogQ:            []int{58, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
// 		LogP:            []int{60, 60, 60},
// 		LogDefaultScale: 46,
// 	})
// 	if err != nil {
// 		t.Fatalf("CKKS参数初始化失败: %v", err)
// 	}

// 	// 随机生成输入矩阵 inputQ, inputK, inputV，范围[-1,1)
// 	inputQ := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
// 	inputK := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
// 	inputV := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
// 	for i := 0; i < modelParams.NumRealRow; i++ {
// 		for j := 0; j < modelParams.NumCol; j++ {
// 			inputQ.Set(i, j, rand.Float64()*4-1)
// 			inputK.Set(i, j, rand.Float64()*4-1)
// 			inputV.Set(i, j, rand.Float64()*4-1)
// 		}
// 	}
// 	// 编码、批处理
// 	inputQ = utils.PadOrTruncateMatrix(inputQ, modelParams)
// 	inputK = utils.PadOrTruncateMatrix(inputK, modelParams)
// 	inputV = utils.PadOrTruncateMatrix(inputV, modelParams)
// 	inputMatsQ := utils.MatrixToBatchMats(inputQ, modelParams)
// 	inputMatsK := utils.MatrixToBatchMats(inputK, modelParams)
// 	inputMatsV := utils.MatrixToBatchMats(inputV, modelParams)
// 	encodedInputsQ := ecdmat.EncodeDense(inputMatsQ, modelParams)
// 	encodedInputsK := ecdmat.EncodeDense(inputMatsK, modelParams)
// 	encodedInputsV := ecdmat.EncodeDense(inputMatsV, modelParams)

// 	// 生成密钥
// 	ecd, enc, eval, dec := he.GenerateKeys(ckksParams, modelParams)

// 	// 加密
// 	ctQ, err := he.EncryptInputMatrices(encodedInputsQ, modelParams, ckksParams, enc, ecd)
// 	if err != nil {
// 		t.Fatalf("加密Q失败: %v", err)
// 	}
// 	ctK, err := he.EncryptInputMatrices(encodedInputsK, modelParams, ckksParams, enc, ecd)
// 	if err != nil {
// 		t.Fatalf("加密K失败: %v", err)
// 	}
// 	ctV, err := he.EncryptInputMatrices(encodedInputsV, modelParams, ckksParams, enc, ecd)
// 	if err != nil {
// 		t.Fatalf("加密K失败: %v", err)
// 	}

// 	// 调用Halevi-Shoup密文矩阵attention主流程
// 	ctExpQKTMulV, ctAdd, err := CiphertextMatricesComputeAttentionWithBSGS(ctQ, ctK, ctV, 0, modelParams, ckksParams, eval)
// 	if err != nil {
// 		t.Fatalf("Halevi-Shoup密文矩阵attention计算失败: %v", err)
// 	}

// 	// 解密
// 	decY, err := he.DecryptCiphertextMatrices(ctExpQKTMulV, modelParams, ckksParams, dec, ecd)
// 	if err != nil {
// 		t.Fatalf("解密失败: %v", err)
// 	}

// 	// 解码
// 	matsY := ecdmat.DecodeDense(decY, modelParams)

// 	// 输出输入矩阵 inputQ
// 	t.Log("输入矩阵Q（inputQ）：")
// 	utils.PrintMat(inputQ)
// 	// 输出输入矩阵 inputK
// 	t.Log("输入矩阵K（inputK）：")
// 	utils.PrintMat(inputK)
// 	// 输出输入矩阵 inputV
// 	t.Log("输入矩阵V（inputV）：")
// 	utils.PrintMat(inputV)

// 	// 明文直接计算 Q × K^T
// 	expectedQK := mat.NewDense(modelParams.NumRow, modelParams.NumRow, nil)
// 	kT := mat.DenseCopyOf(inputK.T())
// 	expectedQK.Product(inputQ, kT)

// 	// 对 expectedQK 做 exp
// 	expMat := mat.NewDense(modelParams.NumRow, modelParams.NumRow, nil)
// 	for i := 0; i < modelParams.NumRow; i++ {
// 		for j := 0; j < modelParams.NumRow; j++ {
// 			val := math.Exp(expectedQK.At(i, j))
// 			expMat.Set(i, j, val)
// 		}
// 	}

// 	// 明文 attention: exp(QK^T) × V
// 	expectedAtt := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
// 	expectedAtt.Product(expMat, inputV)

// 	// 输出明文 QK^T
// 	t.Log("明文 QK^T 结果（expectedQK）：")
// 	utils.PrintMat(expectedQK)
// 	// 输出明文 exp(QK^T)
// 	t.Log("明文 exp(QK^T) 结果：")
// 	utils.PrintMat(expMat)
// 	// 计算明文exp(QK^T)每行的和
// 	rowSums := make([]float64, modelParams.NumRow)
// 	for i := 0; i < modelParams.NumRow; i++ {
// 		sum := 0.0
// 		for j := 0; j < modelParams.NumRow; j++ {
// 			sum += expMat.At(i, j)
// 		}
// 		rowSums[i] = sum
// 	}
// 	t.Logf("明文 exp(QK^T) 每行和: %v", rowSums)
// 	// 输出明文 attention
// 	t.Log("明文 attention exp(QK^T)×V 结果：")
// 	utils.PrintMat(expectedAtt)

// 	// 检查第一个batch的结果与明文计算是否接近
// 	got := matsY[0]
// 	t.Log("密文 attention 结果（Exp(QK^T)×V）：")
// 	utils.PrintMat(got)
// 	ptAdd := he.DecryptCiphertext(ctAdd, ckksParams, dec, ecd)
// 	t.Log("密文计算 Exp(QK^T) 求和结果：")
// 	fmt.Println(ptAdd)
// }

// func TestCiphertextMatricesComputeAttentionWithBSGSAndApproxMax(t *testing.T) {
// 	// 使用小规模模型参数
// 	modelParams := &configs.ModelParams{
// 		NumBatch:    2,
// 		NumRow:      8,
// 		NumCol:      4,
// 		SqrtD:       2.0,
// 		BabyStep:    3,
// 		GiantStep:   3,
// 		ExpSubValue: []float64{0.0},
// 		NumRealRow:  5,
// 	}

// 	// 初始化简单的CKKS参数
// 	ckksParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
// 		LogN:            5,
// 		LogQ:            []int{58, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
// 		LogP:            []int{60, 60, 60},
// 		LogDefaultScale: 46,
// 	})
// 	if err != nil {
// 		t.Fatalf("CKKS参数初始化失败: %v", err)
// 	}

// 	// 随机生成输入矩阵 inputQ, inputK, inputV，范围[-1,1)
// 	inputQ := mat.NewDense(modelParams.NumRealRow, modelParams.NumCol, nil)
// 	inputK := mat.NewDense(modelParams.NumRealRow, modelParams.NumCol, nil)
// 	inputV := mat.NewDense(modelParams.NumRealRow, modelParams.NumCol, nil)
// 	for i := 0; i < modelParams.NumRealRow; i++ {
// 		for j := 0; j < modelParams.NumCol; j++ {
// 			inputQ.Set(i, j, rand.Float64()*6-3)
// 			inputK.Set(i, j, rand.Float64()*6-3)
// 			inputV.Set(i, j, rand.Float64()*6-3)
// 		}
// 	}

// 	// 编码、批处理
// 	inputQ = utils.PadOrTruncateMatrix(inputQ, modelParams)
// 	inputK = utils.PadOrTruncateMatrix(inputK, modelParams)
// 	inputV = utils.PadOrTruncateMatrix(inputV, modelParams)
// 	inputMatsQ := utils.MatrixToBatchMats(inputQ, modelParams)
// 	inputMatsK := utils.MatrixToBatchMats(inputK, modelParams)
// 	inputMatsV := utils.MatrixToBatchMats(inputV, modelParams)
// 	encodedInputsQ := ecdmat.EncodeDense(inputMatsQ, modelParams)
// 	encodedInputsK := ecdmat.EncodeDense(inputMatsK, modelParams)
// 	encodedInputsV := ecdmat.EncodeDense(inputMatsV, modelParams)

// 	// 生成密钥
// 	ecd, enc, eval, dec := he.GenerateKeys(ckksParams, modelParams)

// 	// 加密
// 	ctQ, err := he.EncryptInputMatrices(encodedInputsQ, modelParams, ckksParams, enc, ecd)
// 	if err != nil {
// 		t.Fatalf("加密Q失败: %v", err)
// 	}
// 	ctK, err := he.EncryptInputMatrices(encodedInputsK, modelParams, ckksParams, enc, ecd)
// 	if err != nil {
// 		t.Fatalf("加密K失败: %v", err)
// 	}
// 	ctV, err := he.EncryptInputMatrices(encodedInputsV, modelParams, ckksParams, enc, ecd)
// 	if err != nil {
// 		t.Fatalf("加密K失败: %v", err)
// 	}

// 	// 调用Halevi-Shoup密文矩阵attention主流程
// 	ctExpQKTMulV, ctAdd, err := CiphertextMatricesComputeAttentionWithBSGSAndApproxMax(ctQ, ctK, ctV, modelParams, ckksParams, eval)
// 	if err != nil {
// 		t.Fatalf("Halevi-Shoup密文矩阵attention计算失败: %v", err)
// 	}

// 	// 解密
// 	decY, err := he.DecryptCiphertextMatrices(ctExpQKTMulV, modelParams, ckksParams, dec, ecd)
// 	if err != nil {
// 		t.Fatalf("解密失败: %v", err)
// 	}

// 	// 解码
// 	matsY := ecdmat.DecodeDense(decY, modelParams)

// 	// 输出输入矩阵 inputQ
// 	t.Log("输入矩阵Q（inputQ）：")
// 	utils.PrintMat(inputQ)
// 	// 输出输入矩阵 inputK
// 	t.Log("输入矩阵K（inputK）：")
// 	utils.PrintMat(inputK)
// 	// 输出输入矩阵 inputV
// 	t.Log("输入矩阵V（inputV）：")
// 	utils.PrintMat(inputV)

// 	// 明文直接计算 Q × K^T
// 	expectedQK := mat.NewDense(modelParams.NumRow, modelParams.NumRow, nil)
// 	kT := mat.DenseCopyOf(inputK.T())
// 	expectedQK.Product(inputQ, kT)

// 	// 明文近似最大值计算
// 	rowMeans := make([]float64, modelParams.NumRow)
// 	rowVars := make([]float64, modelParams.NumRow)
// 	rowStds := make([]float64, modelParams.NumRow)
// 	approxMax := make([]float64, modelParams.NumRow)
// 	realRow := float64(modelParams.NumRealRow)
// 	// 明文近似最大值计算（只统计有效token）
// 	for i := 0; i < modelParams.NumRow; i++ {
// 		sum := 0.0
// 		for j := 0; j < int(modelParams.NumRealRow); j++ {
// 			sum += expectedQK.At(i, j)
// 		}
// 		rowMeans[i] = sum / realRow
// 		varSum := 0.0
// 		for j := 0; j < int(modelParams.NumRealRow); j++ {
// 			varSum += math.Pow(expectedQK.At(i, j)-rowMeans[i], 2)
// 		}
// 		rowVars[i] = varSum / realRow
// 		rowStds[i] = 1.25 + 0.1*rowVars[i]
// 		approxMax[i] = rowMeans[i] + rowStds[i]*math.Sqrt(2*math.Log(realRow))
// 		fmt.Println(rowMeans[i], "  ", rowVars[i], "  ", rowStds[i], "  ", varSum)
// 	}
// 	t.Logf("明文 attention 近似最大值: %v", approxMax)

// 	// exp(QK^T - approxMax)，超出部分直接设为0
// 	expMat := mat.NewDense(modelParams.NumRow, modelParams.NumRow, nil)
// 	for i := 0; i < modelParams.NumRow; i++ {
// 		for j := 0; j < modelParams.NumRow; j++ {
// 			if i < int(modelParams.NumRealRow) && j < int(modelParams.NumRealRow) {
// 				val := math.Exp(expectedQK.At(i, j) - approxMax[i])
// 				expMat.Set(i, j, val)
// 			} else {
// 				expMat.Set(i, j, 0) // mask掉无效token（行或列超出都设为0）
// 			}
// 		}
// 	}

// 	// 明文 attention: exp(QK^T - approxMax) × V
// 	expectedAtt := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
// 	expectedAtt.Product(expMat, inputV)

// 	// 输出明文 QK^T
// 	t.Log("明文 QK^T 结果（expectedQK）：")
// 	utils.PrintMat(expectedQK)
// 	// 输出明文 exp(QK^T - approxMax)
// 	t.Log("明文 exp(QK^T - approxMax) 结果：")
// 	utils.PrintMat(expMat)
// 	// 计算明文exp(QK^T - approxMax)每行的和
// 	rowSums := make([]float64, modelParams.NumRow)
// 	for i := 0; i < modelParams.NumRow; i++ {
// 		sum := 0.0
// 		for j := 0; j < modelParams.NumRow; j++ {
// 			sum += expMat.At(i, j)
// 		}
// 		rowSums[i] = sum
// 	}
// 	t.Logf("明文 exp(QK^T - approxMax) 每行和: %v", rowSums)
// 	// 输出明文 attention
// 	t.Log("明文 attention exp(QK^T - approxMax)×V 结果：")
// 	utils.PrintMat(expectedAtt)

// 	// 检查第一个batch的结果与明文计算是否接近
// 	got := matsY[0]
// 	t.Log("密文 attention 结果（Exp(QK^T - approxMax)×V）：")
// 	utils.PrintMat(got)
// 	ptAdd := he.DecryptCiphertext(ctAdd, ckksParams, dec, ecd)
// 	t.Log("密文计算 Exp(QK^T - approxMax) 求和结果：")
// 	fmt.Println(ptAdd)

// 	testMat := utils.GenerateSpecialSquareMatrix(modelParams)
// 	utils.PrintMat(testMat)
// 	diag1 := utils.ExtractAndRepeatDiagonal(testMat, 5, modelParams.NumBatch)
// 	t.Log(diag1)
// 	rotDiag := utils.PlainRotateVec(diag1, 1)
// 	t.Log(rotDiag)
// }

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
