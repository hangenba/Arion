package utils

import (
	"Arion/configs"
	"fmt"
	"math"
	"path/filepath"

	"gonum.org/v1/gonum/mat"
)

type LlamaLayerParameters struct {
	LayerNumber int // 层号，例如 0 表示第一层

	// Attention 部分
	InputRmsNormWeight mat.Matrix // input_rmsnorm_weight.csv， 4096 x 1

	LayerAttentionSelfQueryWeight mat.Matrix // q_proj_weight.csv   4096 x 4096
	LayerAttentionSelfKeyWeight   mat.Matrix // k_proj_weight.csv   1024 x 4096
	LayerAttentionSelfValueWeight mat.Matrix // v_proj_weight.csv   1024 x 4096
	LayerConcateWeight            mat.Matrix // o_proj_weight.csv   4096 x 4096

	// MLP 部分（LLaMA: gate → up → down）
	MLPGateProjWeight mat.Matrix // gate_proj_weight.csv   14336 x 4096
	MLPUpProjWeight   mat.Matrix // up_proj_weight.csv     14336 x 4096
	MLPDownProjWeight mat.Matrix // down_proj_weight.csv   4096 x 14336

	// Output 部分
	PostAttentionRmsNormWeight mat.Matrix // post_attention_rmsnorm_weight.csv   4096 x 1
}

// 测试用，生成 Llama Attention 层参数和输入矩阵
func GenerateLlamaLayerParametersAttention(param *configs.ModelParams) (*LlamaLayerParameters, *mat.Dense) {
	var llamaLayerParams LlamaLayerParameters

	var inputMat *mat.Dense

	llamaLayerParams.LayerAttentionSelfQueryWeight = GenerateRandomMatrixV2(param.NumCol, param.NumCol)
	llamaLayerParams.LayerAttentionSelfKeyWeight = GenerateRandomMatrixV2(param.NumColKV, param.NumCol)
	llamaLayerParams.LayerAttentionSelfValueWeight = GenerateRandomMatrixV2(param.NumColKV, param.NumCol)

	inputMat = GenerateRandomMatrixV2(param.NumRow, param.NumCol)

	return &llamaLayerParams, inputMat
}

// 明文下计算带 RoPE + GQA 的 Self-Attention：
// input:  [seq_len x d_model]  (m x d)
// output: [seq_len x d_model]  (m x d)，32 个 head 拼在一起
func ComputeLlamaAttentionPlain(
	params *LlamaLayerParameters,
	input *mat.Dense,
) *mat.Dense {

	seqLen, dModel := input.Dims()

	const numHeads = 32  // H
	const numKVHeads = 8 // G

	headDim := dModel / numHeads // d_h = 128

	// 1) Q = X * Wq  -> [m x d]
	var Qfull mat.Dense
	Qfull.Mul(input, params.LayerAttentionSelfQueryWeight.T())

	// 2) K = X * Wk  -> [m x (d_h * G)] = [m x 1024]
	var Kfull mat.Dense
	Kfull.Mul(input, params.LayerAttentionSelfKeyWeight.T())

	// 3) V = X * Wv  -> [m x (d_h * G)] = [m x 1024]
	var Vfull mat.Dense
	Vfull.Mul(input, params.LayerAttentionSelfValueWeight.T())

	_, kvCols := Kfull.Dims()
	if kvCols != headDim*numKVHeads {
		panic("ComputeLlamaAttentionPlain: Kfull cols != headDim * numKVHeads")
	}

	// 输出：把 32 个 head 的结果拼成 [m x d]
	outAll := mat.NewDense(seqLen, dModel, nil)

	scale := 1.0 / math.Sqrt(float64(headDim))
	groupSize := numHeads / numKVHeads // 32 / 8 = 4

	// RoPE 的 base，可以按需要调成 Llama3 用的 500000 等，这里先用经典 10000
	const ropeBase = 10000.0

	// 对每个 query head h 做 attention
	for h := 0; h < numHeads; h++ {
		// 所属的 KV group：每 4 个 head 共享 1 个 KV head
		g := h / groupSize // 0..7

		// Q_h: [m x d_h] 从 Qfull 列中切片
		QhMat := Qfull.Slice(0, seqLen, h*headDim, (h+1)*headDim).(*mat.Dense)

		// K_g, V_g: [m x d_h] 从 Kfull/Vfull 列中切片
		KgMat := Kfull.Slice(0, seqLen, g*headDim, (g+1)*headDim).(*mat.Dense)
		VgMat := Vfull.Slice(0, seqLen, g*headDim, (g+1)*headDim) // V 只参与 matmul，不需要 RoPE

		// ★ 在这里对 Q_h, K_g 应用 RoPE（原地修改）
		applyRoPEInPlace(QhMat, KgMat, ropeBase)

		// scores_h = Q_h * K_g^T / sqrt(d_h) -> [m x m]
		var scores mat.Dense
		scores.Mul(QhMat, KgMat.T())
		scores.Scale(scale, &scores)

		// 对每一行做 softmax
		softmaxRowsInPlace(&scores)

		// head 输出 A_h = scores * V_g -> [m x d_h]
		var outH mat.Dense
		outH.Mul(&scores, VgMat)

		// 把 outH 拷贝到 outAll 对应的列区间
		sub := outAll.Slice(0, seqLen, h*headDim, (h+1)*headDim).(*mat.Dense)
		sub.Copy(&outH)
	}

	return outAll
}

// 对 scores 的每一行做 softmax，原地修改
func softmaxRowsInPlace(m *mat.Dense) {
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		row := m.RawRowView(i)

		// 数值稳定：减去 max
		maxVal := row[0]
		for j := 1; j < c; j++ {
			if row[j] > maxVal {
				maxVal = row[j]
			}
		}
		var sum float64
		for j := 0; j < c; j++ {
			row[j] = math.Exp(row[j] - maxVal)
			sum += row[j]
		}
		if sum == 0 {
			continue
		}
		invSum := 1.0 / sum
		for j := 0; j < c; j++ {
			row[j] *= invSum
		}
	}
}

// 对单个 head 的 Q、K 应用 RoPE，原地修改：
// q, k: [seq_len x headDim]
// base: RoPE 频率 base，一般 10000 或更大
func applyRoPEInPlace(q, k *mat.Dense, base float64) {
	seqLen, headDim := q.Dims()
	if headDim%2 != 0 {
		panic("applyRoPEInPlace: headDim must be even")
	}
	half := headDim / 2

	// 预计算 invFreq[i] = base^{-2i/headDim}, i=0..half-1
	invFreq := make([]float64, half)
	d := float64(headDim)
	for i := 0; i < half; i++ {
		invFreq[i] = math.Pow(base, -2.0*float64(i)/d)
	}

	// 对每个位置 p（token index）做二维旋转
	for p := 0; p < seqLen; p++ {
		qRow := q.RawRowView(p)
		kRow := k.RawRowView(p)

		for i := 0; i < half; i++ {
			cosVal := math.Cos(float64(p) * invFreq[i])
			sinVal := math.Sin(float64(p) * invFreq[i])

			// pair indices
			i0 := 2 * i
			i1 := 2*i + 1

			// Q: (x, y) -> (x', y')
			xq := qRow[i0]
			yq := qRow[i1]
			qRow[i0] = xq*cosVal - yq*sinVal
			qRow[i1] = xq*sinVal + yq*cosVal

			// K: 同样旋转
			xk := kRow[i0]
			yk := kRow[i1]
			kRow[i0] = xk*cosVal - yk*sinVal
			kRow[i1] = xk*sinVal + yk*cosVal
		}
	}
}

// ReadLlamaLayerParameters 读取指定层的所有 LLaMA 参数，返回 LlamaLayerParameters
func ReadLlamaLayerParameters(params *configs.ModelParams, layer int) (*LlamaLayerParameters, error) {
	var err error

	// 各子模块参数路径（根据新的目录结构）
	// ModelPath/layer_x/Attention/RMSNorm/param
	inputRmsNormPath := filepath.Join(params.ModelPath,
		fmt.Sprintf("layer_%d", layer),
		"Attention", "RMSNorm", "param")

	// ModelPath/layer_x/Attention/selfattention/param
	selfAttentionPath := filepath.Join(params.ModelPath,
		fmt.Sprintf("layer_%d", layer),
		"Attention", "selfattention", "param")

	// ModelPath/layer_x/MLP/FFN/param
	mlpPath := filepath.Join(params.ModelPath,
		fmt.Sprintf("layer_%d", layer),
		"MLP", "FFN", "param")

	// ModelPath/layer_x/MLP/RMSNorm/param
	postAttentionRmsNormPath := filepath.Join(params.ModelPath,
		fmt.Sprintf("layer_%d", layer),
		"MLP", "RMSNorm", "param")

	lp := &LlamaLayerParameters{
		LayerNumber: layer,
	}

	// ===== InputRMSNorm =====
	lp.InputRmsNormWeight, err = ReadCSVToMatrix(filepath.Join(inputRmsNormPath, "input_rmsnorm_weight.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w",
			filepath.Join(inputRmsNormPath, "input_rmsnorm_weight.csv"), err)
	}

	// ===== Self-Attention: Q/K/V/O 投影 =====
	layerAttentionSelfQueryWeight, err := ReadCSVToMatrix(filepath.Join(selfAttentionPath, "q_proj_weight.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w",
			filepath.Join(selfAttentionPath, "q_proj_weight.csv"), err)
	}
	lp.LayerAttentionSelfQueryWeight = ScaleMatrix(layerAttentionSelfQueryWeight, 1/params.SqrtD)

	lp.LayerAttentionSelfKeyWeight, err = ReadCSVToMatrix(filepath.Join(selfAttentionPath, "k_proj_weight.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w",
			filepath.Join(selfAttentionPath, "k_proj_weight.csv"), err)
	}

	lp.LayerAttentionSelfValueWeight, err = ReadCSVToMatrix(filepath.Join(selfAttentionPath, "v_proj_weight.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w",
			filepath.Join(selfAttentionPath, "v_proj_weight.csv"), err)
	}

	lp.LayerConcateWeight, err = ReadCSVToMatrix(filepath.Join(selfAttentionPath, "o_proj_weight.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w",
			filepath.Join(selfAttentionPath, "o_proj_weight.csv"), err)
	}

	// ===== MLP: gate / up / down =====
	lp.MLPGateProjWeight, err = ReadCSVToMatrix(filepath.Join(mlpPath, "gate_proj_weight.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w",
			filepath.Join(mlpPath, "gate_proj_weight.csv"), err)
	}

	lp.MLPUpProjWeight, err = ReadCSVToMatrix(filepath.Join(mlpPath, "up_proj_weight.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w",
			filepath.Join(mlpPath, "up_proj_weight.csv"), err)
	}

	lp.MLPDownProjWeight, err = ReadCSVToMatrix(filepath.Join(mlpPath, "down_proj_weight.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w",
			filepath.Join(mlpPath, "down_proj_weight.csv"), err)
	}

	// ===== PostAttentionRMSNorm =====
	lp.PostAttentionRmsNormWeight, err = ReadCSVToMatrix(filepath.Join(postAttentionRmsNormPath, "post_attention_rmsnorm_weight.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w",
			filepath.Join(postAttentionRmsNormPath, "post_attention_rmsnorm_weight.csv"), err)
	}

	return lp, nil
}

func PrintLlamaLayerParametersDims(lp *LlamaLayerParameters) {
	printMatrixDims := func(name string, m mat.Matrix) {
		if m == nil {
			fmt.Printf("  %-36s : nil\n", name)
			return
		}
		r, c := m.Dims()
		fmt.Printf("  %-36s : %4d x %-4d\n", name, r, c)
	}

	fmt.Printf("======================= LLaMA Layer %d 参数维度 =======================\n", lp.LayerNumber)

	fmt.Println("[Attention - InputRMSNorm]")
	printMatrixDims("InputRmsNormWeight", lp.InputRmsNormWeight)

	fmt.Println("[Attention - SelfAttention Q/K/V/O]")
	printMatrixDims("LayerAttentionSelfQueryWeight", lp.LayerAttentionSelfQueryWeight)
	printMatrixDims("LayerAttentionSelfKeyWeight", lp.LayerAttentionSelfKeyWeight)
	printMatrixDims("LayerAttentionSelfValueWeight", lp.LayerAttentionSelfValueWeight)
	printMatrixDims("LayerConcateWeight(o_proj)", lp.LayerConcateWeight)

	fmt.Println("[MLP]")
	printMatrixDims("MLPGateProjWeight", lp.MLPGateProjWeight)
	printMatrixDims("MLPUpProjWeight", lp.MLPUpProjWeight)
	printMatrixDims("MLPDownProjWeight", lp.MLPDownProjWeight)

	fmt.Println("[Output - PostAttentionRMSNorm]")
	printMatrixDims("PostAttentionRmsNormWeight", lp.PostAttentionRmsNormWeight)

	fmt.Println("=====================================================================")
}

// ReadInput 根据模型参数中的 ModelPath 拼接文件名并读取 CSV 文件为矩阵
func ReadLlamaInput(params *configs.ModelParams, filename string) (*mat.Dense, error) {
	inputPath := filepath.Join(params.ModelPath, "layer_0", "Attention", "RMSNorm", "allresult", filename)
	return ReadCSVToMatrix(inputPath)
}
