package layernorm

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"Arion/configs"
	"Arion/pkg/ecdmat"
	"Arion/pkg/he"
	"Arion/pkg/utils"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"gonum.org/v1/gonum/mat"
)

// 构造简单的密文矩阵并测试均值和方差
func TestCiphertextMatricesReturnAvgAndVarNorm(t *testing.T) {
	// 1. 初始化参数
	ckksParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            4,
		LogQ:            []int{56, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45},
		LogP:            []int{55, 55, 55, 55},
		LogDefaultScale: 45,
	})
	if err != nil {
		t.Fatalf("CKKS参数初始化失败: %v", err)
	}
	modelParams := &configs.ModelParams{
		NumBatch:         2,
		NumRow:           4,
		NumCol:           768,
		InvSqrtMinValue1: 0.002,
		InvSqrtMaxValue1: 1.4,
		InvSqrtDegree1:   31,
		InvSqrtIter1:     2,
	}

	// 2. 生成密钥
	ecd, enc, eval, dec := he.GenerateKeys(ckksParams, modelParams)

	// 3. 构造明文向量并加密为密文矩阵
	inputQ := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
	for i := 0; i < modelParams.NumRow; i++ {
		for j := 0; j < modelParams.NumCol; j++ {
			inputQ.Set(i, j, rand.Float64()*4-2) // 随机生成范围[-1, 1)的值
		}
	}
	inputQ = utils.PadOrTruncateMatrix(inputQ, modelParams)
	inputMatsQ := utils.MatrixToBatchMats(inputQ, modelParams)
	encodedInputsQ := ecdmat.EncodeDense(inputMatsQ, modelParams)

	ctQ, err := he.EncryptInputMatrices(encodedInputsQ, modelParams, ckksParams, enc, ecd)
	if err != nil {
		t.Fatalf("加密Q失败: %v", err)
	}

	// 4. 调用被测函数
	ctAvg, ctVar, err := CiphertextMatricesReturnAvgAndVarNorm(ctQ, modelParams, ckksParams, eval)
	if err != nil {
		t.Fatalf("CiphertextMatricesReturnAvgAndVarNorm 计算失败: %v", err)
	}

	// 5. 解密并输出
	ptAvg := dec.DecryptNew(ctAvg)
	avg := make([]float64, ckksParams.MaxSlots())
	ecd.Decode(ptAvg, avg)
	t.Logf("密文均值解密: %v", avg[:modelParams.NumBatch*modelParams.NumRow])

	ptVar := dec.DecryptNew(ctVar)
	variance := make([]float64, ckksParams.MaxSlots())
	ecd.Decode(ptVar, variance)
	t.Logf("密文方差解密: %v", variance[:modelParams.NumBatch*modelParams.NumRow])

	// 6. 明文计算均值和方差
	for i := 0; i < modelParams.NumRow; i++ {
		row := inputQ.RawRowView(i)
		sum := 0.0
		for _, v := range row {
			sum += v
		}
		mean := sum / float64(len(row))
		varSum := 0.0
		for _, v := range row {
			varSum += (v - mean) * (v - mean)
		}
		varPlain := varSum / float64(len(row))
		sqrtVarPlain := 1.0 / math.Sqrt(varPlain) // 近似1/sqrt(var)
		t.Logf("第%d行 明文均值: %v, 明文方差: %v, 明文1/sqrt(var): %v", i, mean, varPlain, sqrtVarPlain)
	}
	fmt.Println("计算前的密文层级：", ctQ.Ciphertexts[0].Level())
	fmt.Println("计算后Avg的密文层级：", ctAvg.Level())
	fmt.Println("计算后Var的密文层级：", ctVar.Level())

	// 测试1/sqrt(var)
	ctSqrtVar := InvertSqrtByChebyshevAndNewtonIter(ctVar, &ckksParams, eval, modelParams.InvSqrtMinValue1, modelParams.InvSqrtMaxValue1, modelParams.InvSqrtDegree1, modelParams.InvSqrtIter1)
	ptSqrtVar := dec.DecryptNew(ctSqrtVar)
	sqrtVar := make([]float64, ckksParams.MaxSlots())
	ecd.Decode(ptSqrtVar, sqrtVar)
	t.Logf("密文1/sqrt(var)解密: %v", sqrtVar[:modelParams.NumBatch*modelParams.NumRow])
	fmt.Println("计算后1/sqrt(var)的密文层级：", ctSqrtVar.Level())
}

func TestInvertSqrtByChebyshevAndNewtonIter1(t *testing.T) {
	// 初始化简单的CKKS参数
	ckksParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            4,
		LogQ:            []int{58, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
		LogP:            []int{60, 60, 60},
		LogDefaultScale: 46,
	})
	if err != nil {
		t.Fatalf("CKKS参数初始化失败: %v", err)
	}

	// 构造明文向量（避免0，防止除零）
	values := []float64{2.2 * 128, 3.0 * 128, 4.0 * 128, 4.3 * 128}
	modelParams := &configs.ModelParams{
		NumBatch:         2,
		NumRow:           4,
		NumCol:           768,
		InvSqrtMinValue1: 0.1 * 128,
		InvSqrtMaxValue1: 5 * 128,
		InvSqrtDegree1:   31,
		InvSqrtIter1:     1,
	}

	// 生成密钥
	ecd, enc, eval, dec := he.GenerateKeys(ckksParams, modelParams)

	// 编码并加密
	pt := ckks.NewPlaintext(ckksParams, ckksParams.MaxLevel())
	if err = ecd.Encode(values, pt); err != nil {
		t.Fatalf("编码失败: %v", err)
	}
	ct, err := enc.EncryptNew(pt)
	if err != nil {
		t.Fatalf("加密失败: %v", err)
	}

	// 计算1/sqrt(x)（Chebyshev+Newton迭代近似）
	ctInvSqrt := InvertSqrtByChebyshevAndNewtonIter(
		ct,
		&ckksParams,
		eval,
		modelParams.InvSqrtMinValue1,
		modelParams.InvSqrtMaxValue1,
		modelParams.InvSqrtDegree1,
		modelParams.InvSqrtIter1,
	)

	// 解密
	ptRes := dec.DecryptNew(ctInvSqrt)
	res := make([]float64, len(values))
	ecd.Decode(ptRes, res)

	// 明文计算1/sqrt(x)
	want := make([]float64, len(values))
	for i, v := range values {
		want[i] = 1.0 / math.Sqrt(v)
	}

	// 检查结果
	for i := range values {
		if math.Abs(res[i]-want[i]) > 0.05 {
			t.Errorf("1/sqrt结果不符: got %.4f, want %.4f, diff=%.4f (i=%d)", res[i], want[i], res[i]-want[i], i)
		}
		t.Logf("输入: %.4f, Chebyshev+Newton 1/sqrt: %.4f, 期望: %.4f", values[i], res[i], want[i])
	}
}

func TestInvertSqrtByChebyshevAndNewtonIter2(t *testing.T) {
	// 初始化简单的CKKS参数
	ckksParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            4,
		LogQ:            []int{58, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
		LogP:            []int{60, 60, 60},
		LogDefaultScale: 46,
	})
	if err != nil {
		t.Fatalf("CKKS参数初始化失败: %v", err)
	}

	// 构造明文向量（避免0，防止除零）
	values := []float64{0.701, 0.82, 0.93, 1561.7}
	modelParams := &configs.ModelParams{
		NumBatch:         2,
		NumRow:           4,
		NumCol:           768,
		InvSqrtMinValue1: 0.1,
		InvSqrtMaxValue1: 1562,
		InvSqrtDegree1:   31,
		InvSqrtIter1:     1,
	}

	// 生成密钥
	ecd, enc, eval, dec := he.GenerateKeys(ckksParams, modelParams)

	// 编码并加密
	pt := ckks.NewPlaintext(ckksParams, ckksParams.MaxLevel())
	if err = ecd.Encode(values, pt); err != nil {
		t.Fatalf("编码失败: %v", err)
	}
	ct, err := enc.EncryptNew(pt)
	if err != nil {
		t.Fatalf("加密失败: %v", err)
	}

	// 计算1/sqrt(x)（Chebyshev+Newton迭代近似）
	ctInvSqrt := InvertSqrtByChebyshevAndNewtonIter(
		ct,
		&ckksParams,
		eval,
		modelParams.InvSqrtMinValue1,
		modelParams.InvSqrtMaxValue1,
		modelParams.InvSqrtDegree1,
		modelParams.InvSqrtIter1,
	)

	// 解密
	ptRes := dec.DecryptNew(ctInvSqrt)
	res := make([]float64, len(values))
	ecd.Decode(ptRes, res)

	// 明文计算1/sqrt(x)
	want := make([]float64, len(values))
	for i, v := range values {
		want[i] = 1.0 / math.Sqrt(v)
	}

	// 检查结果
	for i := range values {
		if math.Abs(res[i]-want[i]) > 0.05 {
			t.Errorf("1/sqrt结果不符: got %.4f, want %.4f, diff=%.4f (i=%d)", res[i], want[i], res[i]-want[i], i)
		}
		t.Logf("输入: %.4f, Chebyshev+Newton 1/sqrt: %.4f, 期望: %.4f", values[i], res[i], want[i])
	}
}

func TestLayerNormSelfAttentionOutput(t *testing.T) {
	// 1. 初始化参数
	ckksParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            4,
		LogQ:            []int{56, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45},
		LogP:            []int{55, 55, 55, 55},
		LogDefaultScale: 45,
	})
	if err != nil {
		t.Fatalf("CKKS参数初始化失败: %v", err)
	}
	modelParams := &configs.ModelParams{
		NumBatch:         1,
		NumRow:           4,
		NumCol:           8,
		InvSqrtMinValue1: 0.1,
		InvSqrtMaxValue1: 5,
		InvSqrtDegree1:   31,
		InvSqrtIter1:     2,
	}

	// 2. 生成密钥
	ecd, enc, eval, dec := he.GenerateKeys(ckksParams, modelParams)

	// 3. 构造明文向量并加密为密文矩阵
	inputQ := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
	for i := 0; i < modelParams.NumRow; i++ {
		for j := 0; j < modelParams.NumCol; j++ {
			inputQ.Set(i, j, rand.Float64()*2-1)
		}
	}
	inputQ = utils.PadOrTruncateMatrix(inputQ, modelParams)
	inputMatsQ := utils.MatrixToBatchMats(inputQ, modelParams)
	encodedInputsQ := ecdmat.EncodeDense(inputMatsQ, modelParams)
	ctQ, err := he.EncryptInputMatrices(encodedInputsQ, modelParams, ckksParams, enc, ecd)
	if err != nil {
		t.Fatalf("加密Q失败: %v", err)
	}

	fmt.Println("加密后的密文层级:", ctQ.Ciphertexts[0].Level())

	// 4. 构造gamma和beta
	gamma := mat.NewDense(modelParams.NumCol, 1, nil)
	beta := mat.NewDense(modelParams.NumCol, 1, nil)
	for i := 0; i < modelParams.NumCol; i++ {
		gamma.Set(i, 0, 1.0)
		beta.Set(i, 0, 0.0)
	}

	// 5. 调用被测函数
	ctOut, err := LayerNormSelfAttentionOutput(ctQ, gamma, beta, modelParams, ckksParams, ecd, eval)
	if err != nil {
		t.Fatalf("LayerNormSelfAttentionOutput 计算失败: %v", err)
	}
	fmt.Println("计算后的密文层级:", ctOut.Ciphertexts[0].Level())

	// 6. 解密并输出
	for i, ct := range ctOut.Ciphertexts {
		pt := dec.DecryptNew(ct)
		vals := make([]float64, ckksParams.MaxSlots())
		ecd.Decode(pt, vals)
		t.Logf("LayerNormSelfAttentionOutput 密文解密结果[%d]: %v", i, vals[:modelParams.NumBatch*modelParams.NumRow])
	}
}

func TestLayerNormOutput(t *testing.T) {
	// 1. 初始化参数
	ckksParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            4,
		LogQ:            []int{56, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45},
		LogP:            []int{55, 55, 55, 55},
		LogDefaultScale: 45,
	})
	if err != nil {
		t.Fatalf("CKKS参数初始化失败: %v", err)
	}
	modelParams := &configs.ModelParams{
		NumBatch:         1,
		NumRow:           4,
		NumCol:           8,
		InvSqrtMinValue2: 0.7,
		InvSqrtMaxValue2: 1562,
		InvSqrtDegree2:   31,
		InvSqrtIter2:     2,
	}

	// 2. 生成密钥
	ecd, enc, eval, dec := he.GenerateKeys(ckksParams, modelParams)

	// 3. 构造明文向量并加密为密文矩阵
	inputQ := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
	for i := 0; i < modelParams.NumRow; i++ {
		for j := 0; j < modelParams.NumCol; j++ {
			inputQ.Set(i, j, rand.Float64()*4-2)
		}
	}
	inputQ = utils.PadOrTruncateMatrix(inputQ, modelParams)
	inputMatsQ := utils.MatrixToBatchMats(inputQ, modelParams)
	encodedInputsQ := ecdmat.EncodeDense(inputMatsQ, modelParams)
	ctQ, err := he.EncryptInputMatrices(encodedInputsQ, modelParams, ckksParams, enc, ecd)
	if err != nil {
		t.Fatalf("加密Q失败: %v", err)
	}
	fmt.Println("加密后的密文层级:", ctQ.Ciphertexts[0].Level())

	// 4. 构造gamma和beta
	gamma := mat.NewDense(modelParams.NumCol, 1, nil)
	beta := mat.NewDense(modelParams.NumCol, 1, nil)
	for i := 0; i < modelParams.NumCol; i++ {
		gamma.Set(i, 0, 0.8)
		beta.Set(i, 0, 0.0)
	}

	// 5. 调用被测函数
	ctOut, err := LayerNormOutput(ctQ, gamma, beta, modelParams, ckksParams, ecd, eval)
	if err != nil {
		t.Fatalf("LayerNormOutput 计算失败: %v", err)
	}
	fmt.Println("加密后的密文层级:", ctOut.Ciphertexts[0].Level())

	// 6. 解密并输出
	for i, ct := range ctOut.Ciphertexts {
		pt := dec.DecryptNew(ct)
		vals := make([]float64, ckksParams.MaxSlots())
		ecd.Decode(pt, vals)
		t.Logf("LayerNormOutput 密文解密结果[%d]: %v", i, vals[:modelParams.NumBatch*modelParams.NumRow])
	}
}
