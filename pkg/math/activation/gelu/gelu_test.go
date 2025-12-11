package gelu

import (
	"Arion/configs"
	"Arion/pkg/he"
	"math"
	"testing"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestCiphertextGeluChebyshev(t *testing.T) {
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
	values := []float64{-2.17933846, -1.98915148, -3.03760338, -1.68912685}

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
	ctGelu := CiphertextGeluChebyshev(ct, &ckksParams, eval)

	// 解密
	ptRes := dec.DecryptNew(ctGelu)
	res := make([]float64, len(values))
	ecd.Decode(ptRes, res)

	// 明文GELU计算
	gelu := func(x float64) float64 {
		return 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*math.Pow(x, 3))))
	}
	want := make([]float64, len(values))
	for i, v := range values {
		want[i] = gelu(v)
	}

	// 检查结果
	for i := range values {
		if math.Abs(res[i]-want[i]) > 0.05 {
			t.Errorf("GELU结果不符: got %.4f, want %.4f, diff=%.4f (i=%d)", res[i], want[i], res[i]-want[i], i)
		}
		t.Logf("输入: %.4f, Chebyshev GELU: %.4f, 期望: %.4f", values[i], res[i], want[i])
	}
}
