package plain

import (
	"Arion/pkg/utils"
	"fmt"
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestVerifyKMatrix(t *testing.T) {
	// === 基本参数 ===
	rows, cols := 4, 4096
	outDim := 1024

	// === 加载输入和权重 ===
	x := loadCSVFile(rows, cols, "../../llama3_8b/layer_0/Attention/selfattention/allresult/attn_inputs.csv")
	kWeight := loadCSVFile(outDim, cols, "../../llama3_8b/layer_0/Attention/selfattention/param/k_proj_weight.csv") // (1024,4096)
	kExpected := loadCSVFile(rows, outDim, "../../llama3_8b/layer_0/Attention/selfattention/allresult/K.csv")       // (4,1024)

	// === 计算 K = X * Wk^T ===
	kWeightT := transposeDense(kWeight) // (4096,1024)
	kCalc := mat.NewDense(rows, outDim, nil)
	kCalc.Mul(x, kWeightT)

	// === 对比误差 ===
	diff := mat.NewDense(rows, outDim, nil)
	diff.Sub(kCalc, kExpected)

	mse := 0.0
	maxErr := 0.0
	for i := 0; i < rows; i++ {
		for j := 0; j < outDim; j++ {
			val := diff.At(i, j)
			mse += val * val
			if math.Abs(val) > maxErr {
				maxErr = math.Abs(val)
			}
		}
	}
	mse /= float64(rows * outDim)

	t.Logf("K matrix shape: (%d, %d)", rows, outDim)
	t.Logf("MSE: %.10f", mse)
	t.Logf("Max abs error: %.10f", maxErr)

	if maxErr < 1e-5 {
		t.Log("✅ K 计算正确（仅存在浮点误差）")
	} else {
		t.Errorf("❌ K 不匹配，可能矩阵转置或输入顺序错误")
	}
}

func TestMulArionadAttention(t *testing.T) {
	realRows, rows, cols, numHeads := 5, 128, 768, 12
	x := loadCSVFile(rows, cols, "../../bert_base_data/layer_0/Attention/BertSelfAttention/allresults/embedded_inputs.csv")

	queryWeight := loadCSVFile(768, 768, "../../bert_base_data/layer_0/Attention/BertSelfAttention/parms/query_weight.csv")
	keyWeight := loadCSVFile(768, 768, "../../bert_base_data/layer_0/Attention/BertSelfAttention/parms/key_weight.csv")
	valueWeight := loadCSVFile(768, 768, "../../bert_base_data/layer_0/Attention/BertSelfAttention/parms/value_weight.csv")

	queryWeight = transposeDense(queryWeight)
	keyWeight = transposeDense(keyWeight)
	valueWeight = transposeDense(valueWeight)

	queryBias := loadCSVVec("../../bert_base_data/layer_0/Attention/BertSelfAttention/parms/query_bias.csv")
	keyBias := loadCSVVec("../../bert_base_data/layer_0/Attention/BertSelfAttention/parms/key_bias.csv")
	valueBias := loadCSVVec("../../bert_base_data/layer_0/Attention/BertSelfAttention/parms/value_bias.csv")

	queryMat := LinearTransform(x, queryWeight, queryBias, realRows)
	keyMat := LinearTransform(x, keyWeight, keyBias, realRows)
	valueMat := LinearTransform(x, valueWeight, valueBias, realRows)

	t.Log("Q Matrix :")
	utils.PrintMat(queryMat)

	t.Log("K Matrix :")
	utils.PrintMat(keyMat)

	t.Log("V Matrix :")
	utils.PrintMat(valueMat)

	// attention
	qHeads := splitHeads(queryMat, numHeads)
	kHeads := splitHeads(keyMat, numHeads)
	vHeads := splitHeads(valueMat, numHeads)
	qktTmp := make([]*mat.Dense, numHeads)
	headsOut := make([]*mat.Dense, numHeads)
	for h := 0; h < numHeads; h++ {
		q := qHeads[h]
		k := kHeads[h]
		v := vHeads[h]

		// 1. 计算 Q · K^T
		_, dk := k.Dims()
		kT := mat.DenseCopyOf(k.T())
		var scores mat.Dense
		scores.Mul(q, kT)
		qktTmp[h] = &scores

		// 2. 除以 sqrt(dk)
		scale := 1.0 / math.Sqrt(float64(dk))
		scores.Scale(scale, &scores)

		// 3. 对每一行做 softmax
		r, c := scores.Dims()
		weights := mat.NewDense(r, c, nil)
		for i := 0; i < r; i++ {
			row := scores.RowView(i).(*mat.VecDense)
			max := row.AtVec(0)
			for j := 1; j < c; j++ {
				if v := row.AtVec(j); v > max {
					max = v
				}
			}
			sum := 0.0
			for j := 0; j < c; j++ {
				sum += math.Exp(row.AtVec(j) - max)
			}
			for j := 0; j < c; j++ {
				soft := math.Exp(row.AtVec(j)-max) / sum
				weights.Set(i, j, soft)
			}
		}

		// 4. Attention output = weights · V
		var out mat.Dense
		out.Mul(weights, v)
		headsOut[h] = &out
	}
	resultMat := concatHeads(headsOut)

	qktMat := concatHeads(qktTmp)
	t.Log("QK^T Matrix :")
	utils.PrintMat(qktMat)
	saveCSV("QKT.csv", qktMat)

	t.Log("Result Matrix :")
	utils.PrintMat(resultMat)
}

func TestPlainLayerNorm(t *testing.T) {
	// 读取输入矩阵
	input := loadCSVFile(5, 768, "../../bert_base_data/layer_0/Attention/SelfOutput/allresults/self_output_residual_connection_before_layernorm.csv")
	// 读取权重和偏置
	gamma := loadCSVVec("../../bert_base_data/layer_0/Attention/SelfOutput/parms/self_output_LayerNorm_weight.csv")
	beta := loadCSVVec("../../bert_base_data/layer_0/Attention/SelfOutput/parms/self_output_LayerNorm_bias.csv")

	r, c := input.Dims()
	fmt.Printf("Input Matrix (%d x %d):\n", r, c)
	// fmt.Println("Input Gamma :\n", gamma)
	// fmt.Println("Input Bate :\n", beta)
	// LayerNorm
	eps := 1e-5
	layerNormOut := mat.NewDense(r, c, nil)

	maxVar := math.SmallestNonzeroFloat64
	minVar := math.MaxFloat64
	means := make([]float64, r)
	variances := make([]float64, r)
	for i := 0; i < r; i++ {
		row := input.RawRowView(i)
		mean := 0.0
		for j := 0; j < c; j++ {
			mean += row[j]
		}
		mean /= float64(c)
		means[i] = mean * float64(c)

		variance := 0.0
		for j := 0; j < c; j++ {
			diff := row[j] - mean
			variance += diff * diff
		}
		variance /= float64(c)
		variances[i] = variance
		// 统计最大最小方差
		if variance > maxVar {
			maxVar = variance
		}
		if variance < minVar {
			minVar = variance
		}
		std := math.Sqrt(variance + eps)
		for j := 0; j < c; j++ {
			// y = gamma * (x-mean)/std + beta
			val := gamma.AtVec(j)*(row[j]-mean)/std + beta.AtVec(j)
			layerNormOut.Set(i, j, val)
		}
	}
	fmt.Printf("LayerNorm variance min: %.8f, max: %.8f\n", minVar, maxVar)
	fmt.Printf("Means: %v\n", means)
	fmt.Printf("Variances: %v\n", variances)
	t.Log("LayerNorm Output:")
	utils.PrintMat(layerNormOut)
}

func TestMatricesMaxMin(t *testing.T) {
	// 假设你有多个csv文件
	paths := []string{
		// "../../bert_tiny_data/layer_0/Attention/SelfOutput/allresults/self_output_residual_connection_before_layernorm.csv",
		// "../../bert_tiny_data/layer_1/Attention/SelfOutput/allresults/self_output_residual_connection_before_layernorm.csv",
		"../../bert_tiny_data/layer_0/Output/allresults/final_output_residual_connection_before_layernorm.csv",
		"../../bert_tiny_data/layer_1/Output/allresults/final_output_residual_connection_before_layernorm.csv",
	}
	mats := make([]*mat.Dense, 0, len(paths))
	for _, p := range paths {
		m := loadCSVFile(5, 128, p)
		mats = append(mats, m)
	}

	minVar := math.MaxFloat64
	maxVar := math.SmallestNonzeroFloat64
	for _, m := range mats {
		r, c := m.Dims()
		for i := 0; i < r; i++ {
			row := m.RawRowView(i)
			mean := 0.0
			for j := 0; j < c; j++ {
				mean += row[j]
			}
			mean /= float64(c)
			variance := 0.0
			for j := 0; j < c; j++ {
				diff := row[j] - mean
				variance += diff * diff
			}
			variance /= float64(c)
			if variance > maxVar {
				maxVar = variance
			}
			if variance < minVar {
				minVar = variance
			}
		}
	}
	fmt.Printf("所有矩阵所有行方差的最小值: %.8f, 最大值: %.8f\n", minVar, maxVar)
}
