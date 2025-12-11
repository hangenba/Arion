package softmax

import (
	"Arion/configs"
	"Arion/pkg/ecdmat"
	"Arion/pkg/he"
	"Arion/pkg/utils"
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"gonum.org/v1/gonum/mat"
)

func TestCiphertextMatricesExp(t *testing.T) {
	// 使用小规模模型参数
	modelParams := &configs.ModelParams{
		NumBatch: 2,
		NumRow:   2,
		NumCol:   2,
		SqrtD:    2.0,
	}

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

	// 构造有规律的输入矩阵 input
	// input := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
	// for i := 0; i < modelParams.NumRow; i++ {
	// 	for j := 0; j < modelParams.NumCol; j++ {
	// 		input.Set(i, j, float64(i*modelParams.NumCol+j+1)) // 1,2,3,4
	// 	}
	// }

	// 显式指定输入矩阵的值
	inputData := []float64{
		10, -10,
		-60, 15,
	}
	input := mat.NewDense(modelParams.NumRow, modelParams.NumCol, inputData)
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

	// 计算exp
	startNewton := time.Now()
	var ctExp *he.CiphertextMatrices
	for i := 0; i < 10; i++ {
		ctExp = CiphertextMatricesExpChebyshev(ctX, &ckksParams, eval)
	}
	elapsedNewton := time.Since(startNewton).Seconds() / 10.0

	// 解密
	decY, err := he.DecryptCiphertextMatrices(ctExp, modelParams, ckksParams, dec, ecd)
	if err != nil {
		t.Fatalf("解密失败: %v", err)
	}

	// 解码
	matsY := ecdmat.DecodeDense(decY, modelParams)

	// 输出输入矩阵 input
	t.Log("输入矩阵（input）：")
	utils.PrintMat(input)

	// 输出真实exp函数的结果
	realExp := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
	for i := 0; i < modelParams.NumRow; i++ {
		for j := 0; j < modelParams.NumCol; j++ {
			x := input.At(i, j)
			realExp.Set(i, j, math.Exp(x))
		}
	}
	t.Log("真实exp函数结果：")
	utils.PrintMat(realExp)

	// 明文直接计算exp近似
	expected := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
	for i := 0; i < modelParams.NumRow; i++ {
		for j := 0; j < modelParams.NumCol; j++ {
			x := input.At(i, j)
			// exp(x) ≈ (x*0.0078125+1)^128
			y := math.Pow(x*0.0078125+1, 128)
			expected.Set(i, j, y)
		}
	}

	// 输出明文计算结果
	t.Log("明文exp近似结果（expected）：")
	utils.PrintMat(expected)

	// 检查第一个batch的结果与明文计算是否接近
	got := matsY[0]
	rows, cols := got.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			want := expected.At(i, j)
			val := got.At(i, j)
			if diff := math.Abs(want - val); diff > 1e-1 {
				t.Errorf("结果不符: got %.4f, want %.4f, diff=%.4f (i=%d, j=%d)", val, want, diff, i, j)
			}
		}
	}
	// 输出明文计算结果
	t.Log("密文切比雪夫exp近似结果（compute）：")
	utils.PrintMat(got)

	fmt.Println("原始的密文层级：", ctX.Ciphertexts[0].Level())
	fmt.Println("牛顿迭代运算后的密文层级：", ctExp.Ciphertexts[0].Level())
	fmt.Printf("牛顿迭代密文exp平均耗时: %.6f 秒\n", elapsedNewton)

	// ========== 新增：测试 Chebyshev 多项式近似 exp ==========
	startCheb := time.Now()
	var ctExpCheb *he.CiphertextMatrices
	for i := 0; i < 10; i++ {
		ctExpCheb = CiphertextMatricesExpChebyshev(ctX, &ckksParams, eval)
	}
	elapsedCheb := time.Since(startCheb).Seconds() / 10.0
	decYCheb, err := he.DecryptCiphertextMatrices(ctExpCheb, modelParams, ckksParams, dec, ecd)
	if err != nil {
		t.Fatalf("Chebyshev解密失败: %v", err)
	}
	matsYCheb := ecdmat.DecodeDense(decYCheb, modelParams)
	gotCheb := matsYCheb[0]
	t.Log("Chebyshev多项式近似exp的密文解密结果：")
	utils.PrintMat(gotCheb)

	fmt.Println("原始的密文层级：", ctX.Ciphertexts[0].Level())
	fmt.Println("切比雪夫近似运算后的密文层级：", ctExpCheb.Ciphertexts[0].Level())
	fmt.Printf("切比雪夫密文exp平均耗时: %.6f 秒\n", elapsedCheb)
}

func TestCiphertextMatricesInverse(t *testing.T) {
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

	// 构造明文向量
	values := []float64{0.039, 1.0, 0.623, 242}
	modelParams := &configs.ModelParams{
		NumBatch:    1,
		NumRow:      1,
		NumCol:      len(values),
		SqrtD:       2.0,
		InvMinValue: 0.035,
		InvMaxValue: 243.0,
		InvDegree:   227,
		InvIter:     2,
	}

	// 生成密钥
	ecd, enc, eval, dec := he.GenerateKeys(ckksParams, modelParams)

	// 编码并加密
	pt := ckks.NewPlaintext(ckksParams, ckksParams.MaxLevel())
	if err = ecd.Encode(values, pt); err != nil {
		panic(err)
	}
	ct, err := enc.EncryptNew(pt)
	if err != nil {
		panic(err)
	}

	// 计算逆
	fmt.Println("原始的密文层级：", ct.Level())
	// ctInv := CiphertextMatricesInverseGoldschmidt(ct, iter, 1, 1, &ckksParams, ecd, enc, eval)
	ctInv := CiphertextInverse(ct, &ckksParams, eval, modelParams.InvMinValue, modelParams.InvMaxValue, modelParams.InvDegree, modelParams.InvIter)
	// ctInv := CiphertextInverseChebyshev(ct, &ckksParams, eval)
	fmt.Println("计算后密文层级：", ctInv.Level())

	// 解密
	ptRes := dec.DecryptNew(ctInv)
	res := make([]float64, len(values))
	ecd.Decode(ptRes, res)

	// 检查结果
	for i, v := range values {
		want := 1.0 / v
		got := res[i]
		if math.Abs(got-want) > 0.05 {
			t.Errorf("倒数结果不符: got %.4f, want %.4f, diff=%.4f (i=%d)", got, want, got-want, i)
		}
		t.Logf("输入: %.4f, 倒数: %.4f, 期望: %.4f", v, got, want)
	}
	fmt.Println("Inputs:", values)
	fmt.Println("Ciphertext Inverses:", res[0:4])
}

func TestCiphertextInverseChebyshev(t *testing.T) {
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
	values := []float64{0.039, 1.0, 0.623, 242}
	modelParams := &configs.ModelParams{
		NumBatch:    1,
		NumRow:      1,
		NumCol:      len(values),
		SqrtD:       2.0,
		InvMinValue: 0.035,
		InvMaxValue: 243.0,
		InvDegree:   227,
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

	// 计算倒数（Chebyshev多项式近似）
	ctInv := CiphertextInverseChebyshev(ct, &ckksParams, eval, modelParams.InvMinValue, modelParams.InvMaxValue, modelParams.InvDegree)

	// 解密
	ptRes := dec.DecryptNew(ctInv)
	res := make([]float64, len(values))
	ecd.Decode(ptRes, res)

	// 明文计算倒数
	want := make([]float64, len(values))
	for i, v := range values {
		want[i] = 1.0 / v
	}

	// 检查结果
	for i := range values {
		if math.Abs(res[i]-want[i]) > 0.05 {
			t.Errorf("倒数结果不符: got %.4f, want %.4f, diff=%.4f (i=%d)", res[i], want[i], res[i]-want[i], i)
		}
		t.Logf("输入: %.4f, Chebyshev倒数: %.4f, 期望: %.4f", values[i], res[i], want[i])
	}
}
