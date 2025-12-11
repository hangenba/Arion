package tanh

import (
	"Arion/configs"
	"Arion/pkg/ecdmat"
	"Arion/pkg/he"
	"Arion/pkg/utils"
	"math"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"gonum.org/v1/gonum/mat"
)

func TestCiphertextTanhChebyshev(t *testing.T) {
	// 初始化CKKS参数
	ckksParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            4,
		LogQ:            []int{58, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
		LogP:            []int{60, 60, 60},
		LogDefaultScale: 46,
	})
	if err != nil {
		t.Fatalf("CKKS参数初始化失败: %v", err)
	}
	modelParams := &configs.ModelParams{
		NumBatch: 1,
		NumRow:   1,
		NumCol:   4,
		SqrtD:    2.0,
	}

	// 构造明文向量
	values := []float64{-2.0, -0.5, 0.0, 2.0}

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

	// 计算GELU（Chebyshev多项式近似）
	ctGelu := CiphertextTanhChebyshev(ct, &ckksParams, eval)

	// 解密
	ptRes := dec.DecryptNew(ctGelu)
	res := make([]float64, len(values))
	ecd.Decode(ptRes, res)

	// 明文GELU计算
	tanh := func(x float64) float64 {
		return math.Tanh(x)
	}
	want := make([]float64, len(values))
	for i, v := range values {
		want[i] = tanh(v)
	}

	// 检查结果
	for i := range values {
		if math.Abs(res[i]-want[i]) > 0.05 {
			t.Errorf("GELU结果不符: got %.4f, want %.4f, diff=%.4f (i=%d)", res[i], want[i], res[i]-want[i], i)
		}
		t.Logf("输入: %.4f, Chebyshev GELU: %.4f, 期望: %.4f", values[i], res[i], want[i])
	}
}

func TestCiphertextMatricesOutputLinear(t *testing.T) {
	// 设置参数
	logN := 4
	slots := 1 << (logN - 1) // 16
	numBatch := 2
	numRow := 4 // 2*8=16，刚好填满slots
	numCol := 8

	ckksParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            logN,
		LogQ:            []int{58, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
		LogP:            []int{60, 60, 60},
		LogDefaultScale: 46,
	})
	if err != nil {
		t.Fatalf("CKKS参数初始化失败: %v", err)
	}
	modelParams := &configs.ModelParams{
		NumBatch: numBatch,
		NumRow:   numRow,
		NumCol:   numCol,
	}

	ecd, enc, eval, dec := he.GenerateKeys(ckksParams, modelParams)

	// 构造明文矩阵，每列一个向量
	ctMats := &he.CiphertextMatrices{
		Ciphertexts: make([]*rlwe.Ciphertext, numCol),
		NumBatch:    numBatch,
		NumRow:      numRow,
		NumCol:      numCol,
	}
	for i := 0; i < numCol; i++ {
		// 构造长度为slots的向量，前numBatch*numRow填充数据，剩余填0
		vec := make([]float64, slots)
		for j := 0; j < numBatch*numRow; j++ {
			vec[j] = float64(i+1) + float64(j) // 可自定义
		}
		pt := ckks.NewPlaintext(ckksParams, ckksParams.MaxLevel())
		if err := ecd.Encode(vec, pt); err != nil {
			t.Fatalf("编码失败: %v", err)
		}
		ct, err := enc.EncryptNew(pt)
		if err != nil {
			t.Fatalf("加密失败: %v", err)
		}
		ctMats.Ciphertexts[i] = ct

		// 加密后立即解密，输出明文
		ptDec := dec.DecryptNew(ct)
		decVec := make([]float64, slots)
		ecd.Decode(ptDec, decVec)
		t.Logf("加密后立即解密，第%d列: %v", i, decVec[:numBatch*numRow])
	}

	// 构造明文权重矩阵（如4x2）
	weight := make([][]float64, numCol)
	for i := 0; i < numCol; i++ {
		weight[i] = make([]float64, 2)
		for j := 0; j < 2; j++ {
			weight[i][j] = float64(i + j + 1)
		}
	}
	// 转为 mat.Matrix
	ptMat := mat.NewDense(numCol, 2, nil)
	for i := 0; i < numCol; i++ {
		for j := 0; j < 2; j++ {
			ptMat.Set(i, j, weight[i][j])
		}
	}

	// 调用被测函数
	ctRes, err := CiphertextMatricesOutputLinear(ctMats, modelParams, ckksParams, eval)
	if err != nil {
		t.Fatalf("函数执行失败: %v", err)
	}

	// 解密并输出结果
	for i, ct := range ctRes.Ciphertexts {
		pt := dec.DecryptNew(ct)
		res := make([]float64, slots)
		ecd.Decode(pt, res)
		t.Logf("输出密文[%d]解密结果: %v", i, res[:numBatch*numRow])
	}

	ptRes, err := he.DecryptCiphertextMatrices(ctRes, modelParams, ckksParams, dec, ecd)
	if err != nil {
		t.Fatalf("解密失败: %v", err)
	}

	ptDecRes := ecdmat.DecodeOutputDenseToMatrix(ptRes, modelParams)
	t.Logf("解码后的输出矩阵:\n")
	utils.PrintMat(ptDecRes)

}
