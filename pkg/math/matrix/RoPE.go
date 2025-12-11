package matrix

import (
	"Arion/configs"
	"Arion/pkg/he"
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// ComputeRoPE 对输入的 qk 矩阵应用 Rotary Positional Embedding (RoPE) 变换。
// 输入说明：
//   - qk: 输入矩阵，形状为 (seqLen, dim)（也可视为 (batch*seqLen, dim)）
//   - rotaryDim: 对前 rotaryDim 个维度应用 RoPE（必须为偶数且 <= dim）
//   - base: RoPE 中的基数，常用 10000.0（用于计算频率 inv_freq = base^(−2i/rotaryDim)）
// 返回：
//   - 一个新的 *mat.Dense，包含应用 RoPE 后的结果（不修改原 qk）
//   - 当参数不合法时返回 error
//
// 变换细节（简要）：
//   RoPE 将特征向量的前 rotaryDim 维按照偶数/奇数两两一组进行旋转。
//   对于第 i 对 (x_{2i}, x_{2i+1})，在序列位置 pos 上，计算角度 angle = pos * inv_freq[i]，
//   新的值为：
//     x'_{2i}   =  x_{2i} * cos(angle) - x_{2i+1} * sin(angle)
//     x'_{2i+1} =  x_{2i} * sin(angle) + x_{2i+1} * cos(angle)
//   其中 inv_freq[i] = base^( -2*i / rotaryDim )。
// 注意：
//   - 该实现把行索引视为位置 pos（pos 从 0 到 seqLen-1），如果你的 qk 包含 batch 维度且按行堆叠多个序列，
//     请确保传入的 seqLen 与行索引的语义一致，或在调用前分割 batch。
//   - rotaryDim 应为偶数。
//   - 该实现对剩余维度（>= rotaryDim）保持不变。

func CiphertextMatricesComputeRotaryPositionEmbedding(ctQ []*he.CiphertextMatrices, modelParam *configs.ModelParams, ckksParams ckks.Parameters, eval *ckks.Evaluator) []*he.CiphertextMatrices {
	rotaryDim := ctQ[0].NumCol
	batches := modelParam.NumBatch
	seqLen := modelParam.NumRow
	fmt.Println(rotaryDim, seqLen, batches)
	vecCos, vecSin := GenRoPEBroadcast(rotaryDim, seqLen, batches, 10000.0)

	if rotaryDim%2 != 0 {
		panic("CiphertextMatricesComputeRotaryPositionEmbedding: rotaryDim must be even")
	}

	// utils.PrintVector(vecCos[0])
	// utils.PrintVector(vecSin[0])
	for i := 0; i < len(ctQ); i++ {
		// fmt.Println(i)
		for j := 0; j < rotaryDim/2; j++ {

			// 计算奇数的结果
			ctNewQ1_one, err := eval.MulNew(ctQ[i].Ciphertexts[j*2], vecCos[j*2])
			if err != nil {
				panic(err)
			}
			ctNewQ1_two, err := eval.MulNew(ctQ[i].Ciphertexts[j*2+1], vecSin[j*2])
			if err != nil {
				panic(err)
			}

			// 计算偶数的结果
			ctNewQ2_one, err := eval.MulNew(ctQ[i].Ciphertexts[j*2], vecSin[j*2+1])
			if err != nil {
				panic(err)
			}
			ctNewQ2_two, err := eval.MulNew(ctQ[i].Ciphertexts[j*2+1], vecCos[j*2+1])
			if err != nil {
				panic(err)
			}
			// 合并结果
			ctNewQ1, err := eval.AddNew(ctNewQ1_one, ctNewQ1_two)
			if err != nil {
				panic(err)
			}
			eval.Rescale(ctNewQ1, ctNewQ1)
			ctQ[i].Ciphertexts[j*2] = ctNewQ1

			ctNewQ2, err := eval.AddNew(ctNewQ2_one, ctNewQ2_two)
			if err != nil {
				panic(err)
			}
			eval.Rescale(ctNewQ2, ctNewQ2)
			ctQ[i].Ciphertexts[j*2+1] = ctNewQ2
		}

	}
	return ctQ
}

func GenRoPEBroadcast(d, n, batches int, base float64) ([][]float64, [][]float64) {
	if d%2 != 0 {
		panic("GenRoPEBroadcast: head dim d must be even")
	}
	if base <= 0 {
		base = 50000.0
	}

	// 1) 预计算频率 invFreq[k] = base^{-2k/d}
	half := d / 2
	invFreq := make([]float64, half)
	df := float64(d)
	for k := 0; k < half; k++ {
		invFreq[k] = math.Pow(base, -2.0*float64(k)/df)
	}

	// 2) 先得到 cos[d][n], sin[d][n]（外层维度、内层位置）
	cos := make([][]float64, d)
	sin := make([][]float64, d)
	for j := 0; j < d; j++ {
		cos[j] = make([]float64, n)
		sin[j] = make([]float64, n)
	}
	for pos := 0; pos < n; pos++ {
		p := float64(pos)
		for k := 0; k < half; k++ {
			theta := p * invFreq[k]
			c := math.Cos(theta)
			s := math.Sin(theta)
			j0 := 2 * k
			j1 := 2*k + 1
			cos[j0][pos], cos[j1][pos] = c, c
			sin[j0][pos], sin[j1][pos] = -s, s
		}
	}
	// utils.PrintVector(cos[2])
	// utils.PrintVector(sin[2])

	// 3) 广播到 CKKS 槽位：vec? [d][n*batches]，idx = j*batches + k
	slots := n * batches
	vecCos := make([][]float64, d)
	vecSin := make([][]float64, d)
	for i := 0; i < d; i++ {
		rowCos := make([]float64, slots)
		rowSin := make([]float64, slots)
		for j := 0; j < n; j++ {
			c := cos[i][j]
			s := sin[i][j]
			for k := 0; k < batches; k++ {
				idx := j*batches + k
				rowCos[idx] = c
				rowSin[idx] = s
			}
		}
		vecCos[i], vecSin[i] = rowCos, rowSin
	}

	return vecCos, vecSin
}
