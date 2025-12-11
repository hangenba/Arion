package utils

import (
	"Arion/configs"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"strings"

	"gonum.org/v1/gonum/mat"
)

/*
 * This package includes utility functions for matrix and vector operations
 * 1. PrintMat: prints a matrix in a readable format
 * 2. PrintVector: prints a vector, omitting elements in the middle if its length exceeds 6
 * 3. GenerateRandomMatrix: generates a random matrix with specified dimensions
 * 4. GenerateMultipleRandomMatrices: generates multiple random matrices with specified dimensions
 * 5. ExtractDiagonalElements: extracts 'length' diagonal elements from a given starting position (i, j) in the matrix
 * 6. GCD: calculates the greatest common divisor using the Euclidean algorithm
 * 7. extendedGCD: computes the greatest common divisor of a and b using the extended Euclidean algorithm
 * 8. ModInverse: computes the modular inverse of a under modulo m using the extended Euclidean algorithm
 * 9. Mod: ensures the result is always non-negative
 * 10. MaxDifference: calculates the maximum absolute difference between two matrices
 * 11. RotateSliceNew: returns a new slice that is a rotated version of the input slice
 */

// MatPrint prints a matrix in a readable format.
// If the matrix is larger than 8x8, it displays a truncated view.
func PrintMat(X mat.Matrix) {
	r, c := X.Dims()

	threshold := 8
	if r > threshold || c > threshold {
		// We only want to print parts of the matrix if it is too large
		// fmt.Printf("Matrix (%dx%d) exceeds %dx%d threshold, displaying partial view:\n", r, c, threshold, threshold)
		printPartialMatrix(X, r, c, threshold)
	} else {
		// Use full formatted print if within threshold
		fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
		fmt.Printf("%v\n", fa)
	}
}

// PrintVector prints a vector, omitting elements in the middle if its length exceeds 6.
func PrintVector(v []float64) {
	n := len(v)
	if n <= 6 {
		fmt.Println(v)
		return
	}

	// Create a string representation with ellipsis for vectors longer than 6
	var sb strings.Builder
	sb.WriteString("[")
	for i := 0; i < 3; i++ {
		sb.WriteString(fmt.Sprintf("%v, ", v[i]))
	}
	sb.WriteString("..., ")
	for i := n - 3; i < n; i++ {
		sb.WriteString(fmt.Sprintf("%v", v[i]))
		if i < n-1 {
			sb.WriteString(", ")
		}
	}
	sb.WriteString("]")

	fmt.Println(sb.String())
}

// printPartialMatrix prints a truncated version of a matrix.
func printPartialMatrix(X mat.Matrix, r, c, threshold int) {
	for i := 0; i < r; i++ {
		fmt.Print("[")
		if i == 3 && r > threshold {
			fmt.Println("...]")
			i = r - 3
		}

		for j := 0; j < c; j++ {
			if j == 3 && c > threshold {
				fmt.Print("... ")
				j = c - 3
			}

			fmt.Printf("%6.4f ", X.At(i, j))
		}
		fmt.Println("]")
	}
}

// GenerateRandomMatrix generates a random matrix with specified dimensions.
func GenerateRandomMatrix(rows, cols int) *mat.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rand.Float64() / math.Pow(2, 5) // Generate a random float64 number.
	}
	return mat.NewDense(rows, cols, data)
}

func GenerateRandomMatrixV2(rows, cols int) *mat.Dense {
	data := make([]float64, rows*cols)
	scale := math.Pow(2, -5) // 1/32
	for i := range data {
		// rand.Float64() in [0,1) -> (rand*2-1) in (-1,1)
		data[i] = (rand.Float64()*2.0 - 1.0) * scale
	}
	return mat.NewDense(rows, cols, data)
}

// GenerateMultipleRandomMatrices generates multiple random matrices with specified dimensions.
func GenerateMultipleRandomMatrices(numMatrices, rows, cols int) []*mat.Dense {
	matrices := make([]*mat.Dense, numMatrices)
	for i := 0; i < numMatrices; i++ {
		matrices[i] = GenerateRandomMatrix(rows, cols)
	}
	return matrices
}

// ExtractDiagonalElements extracts 'length' diagonal elements
// from a given starting position (i, j) in the matrix, using modulo
// to wrap around when exceeding matrix boundaries.
func ExtractDiagonalElements(X *mat.Dense, i, j, length int) ([]float64, error) {
	r, c := X.Dims()

	// Check if the length is valid
	if length < 0 {
		return nil, errors.New("length must be non-negative")
	}

	// Use Mod function to handle negative indices
	startRow := Mod(i, r)
	startCol := Mod(j, c)

	// Prepare a slice to store extracted elements
	elements := make([]float64, 0, length)

	for k := 0; k < length; k++ {
		// Calculate the current row and column using modulo to wrap around
		row := Mod(startRow+k, r)
		col := Mod(startCol+k, c)

		// Append the element to the list
		elements = append(elements, X.At(row, col))
	}

	return elements, nil
}

// GCD calculates the greatest common divisor using the Euclidean algorithm
func GCD(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// extendedGCD computes the greatest common divisor of a and b using the
// extended Euclidean algorithm. It returns gcd, x, y where gcd is the
// greatest common divisor and x, y are the coefficients satisfying
// the equation: a*x + b*y = gcd.
func extendedGCD(a, b int) (int, int, int) {
	if b == 0 {
		return a, 1, 0
	}
	gcd, x1, y1 := extendedGCD(b, a%b)
	x := y1
	y := x1 - (a/b)*y1
	return gcd, x, y
}

// modInverse computes the modular inverse of a under modulo m using the
// extended Euclidean algorithm. It returns an error if the modular inverse
// does not exist (i.e., a and m are not coprime).
func ModInverse(a, m int) (int, error) {
	gcd, x, _ := extendedGCD(a, m)
	if gcd != 1 {
		return 0, errors.New("modular inverse does not exist")
	}
	// x might be negative, so we take the result mod m to ensure it is positive.
	return (x%m + m) % m, nil
}

// mod function ensures the result is always non-negative
func Mod(a, b int) int {
	result := a % b
	if result < 0 {
		result += b
	}
	return result
}

// MaxDifference calculates the maximum absolute difference between two matrices.
func MaxDifference(a, b *mat.Dense) error {
	r1, c1 := a.Dims()
	r2, c2 := b.Dims()

	// Ensure matrices have the same dimensions
	if r1 != r2 || c1 != c2 {
		return fmt.Errorf("matrices must have the same dimensions")
	}

	maxDiff := 0.0

	// Iterate over each element to find the maximum difference
	for i := 0; i < r1; i++ {
		for j := 0; j < c1; j++ {
			diff := math.Abs(a.At(i, j) - b.At(i, j))
			if diff > maxDiff {
				maxDiff = diff
			}
		}
	}
	fmt.Printf("Largest err:|%f|\n", maxDiff)

	return nil
}

// RotatedSlice returns a new slice that is a rotated version of the input slice
func RotateSliceNew(s []float64, k int) []float64 {
	n := len(s)
	if n == 0 {
		return nil
	}
	k = k % n
	if k < 0 {
		k += n
	}
	// Create a new slice with the rotated elements
	return append(s[k:], s[:k]...)
}

// PadOrTruncateMatrix 将输入矩阵调整为 modelParams.NumRow × modelParams.NumCol 的大小，超出部分截断，不足部分补零
func PadOrTruncateMatrix(m *mat.Dense, params *configs.ModelParams) *mat.Dense {
	targetRows := params.NumRow
	targetCols := params.NumCol
	rows, cols := m.Dims()
	if cols > targetCols {
		targetCols = cols
	}
	data := make([]float64, targetRows*targetCols)
	for i := 0; i < targetRows; i++ {
		for j := 0; j < targetCols; j++ {
			if i < rows && j < cols {
				data[i*targetCols+j] = m.At(i, j)
			} else {
				data[i*targetCols+j] = 0
			}
		}
	}
	params.NumRealRow = rows // 更新实际行数
	return mat.NewDense(targetRows, targetCols, data)
}

// ScaleMatrix 返回一个新的矩阵，其元素为原矩阵每个元素乘以 scale
func ScaleMatrix(m *mat.Dense, scale float64) *mat.Dense {
	rows, cols := m.Dims()
	result := mat.NewDense(rows, cols, nil)
	result.Scale(scale, m)
	return result
}

// ExpandMatricesToBatch 将输入的 []*mat.Dense 扩展或截断为 numBatch 个，顺序复制已有内容
func ExpandMatricesToBatch(mats []*mat.Dense, numBatch int) []*mat.Dense {
	result := make([]*mat.Dense, numBatch)
	n := len(mats)
	for i := 0; i < numBatch; i++ {
		src := mats[i%n]
		rows, cols := src.Dims()
		data := make([]float64, rows*cols)
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				data[r*cols+c] = src.At(r, c)
			}
		}
		result[i] = mat.NewDense(rows, cols, data)
	}
	return result
}

// MatrixToBatchMats 先将单个 mat.Dense 转为 []*mat.Dense（长度为1），再扩展到 numBatch 个
func MatrixToBatchMats(m *mat.Dense, params *configs.ModelParams) []*mat.Dense {
	return ExpandMatricesToBatch([]*mat.Dense{m}, params.NumBatch)
}

// PrintFirstOfVectors 打印二维数组每个向量的前n位元素
func PrintFirstOfVectors(vecs [][]float64, number int) {
	for i, vec := range vecs {
		fmt.Printf("Vector %d: ", i)
		n := number
		if len(vec) < n {
			n = len(vec)
		}
		for j := 0; j < n; j++ {
			fmt.Printf("%.4f ", vec[j])
		}
		fmt.Println()
	}
}

// GenerateSpecialSquareMatrix 生成一个方阵，大小为 params.NumRow × params.NumRow，
// 前 params.NumRealRow 行和列为 0，超出部分为 -60
func GenerateSpecialSquareMatrix(params *configs.ModelParams) *mat.Dense {
	n := params.NumRow
	real := params.NumRealRow
	data := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i < real && j < real {
				data[i*n+j] = 1
			} else {
				data[i*n+j] = 0
			}
		}
	}
	return mat.NewDense(n, n, data)
}

func GenerateSpecialSquareMatrixLlama(params *configs.ModelParams) *mat.Dense {
	n := params.NumRow        // 矩阵总行列数
	real := params.NumRealRow // 有效的前 real 行/列

	data := make([]float64, n*n)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			// 只在前 real×real 子矩阵内，并且是下三角（含对角线）的地方设为 1
			if i < real && j < real && j <= i {
				data[i*n+j] = 1.0
			} else {
				data[i*n+j] = 0.0
			}
		}
	}

	return mat.NewDense(n, n, data)
}

// ExtractAndRepeatDiagonal 提取方阵 m 的第 diagIdx 条对角线，并将每个元素重复 repeat 次
// 返回长度为 len(diag)*repeat 的切片
// ExtractAndRepeatDiagonal 提取方阵 m 的第 diagIdx 条广义对角线（超出cols则模cols），并将每个元素重复 repeat 次
// 返回长度为 len(diag)*repeat 的切片
func ExtractAndRepeatDiagonal(m *mat.Dense, diagIdx, repeat int) []float64 {
	rows, cols := m.Dims()
	n := rows
	if cols < n {
		n = cols
	}
	diag := make([]float64, 0, n)
	for i := 0; i < n; i++ {
		j := (i + diagIdx) % cols
		if j < 0 {
			j += cols
		}
		diag = append(diag, m.At(i, j))
	}
	// 重复
	result := make([]float64, 0, len(diag)*repeat)
	for _, v := range diag {
		for k := 0; k < repeat; k++ {
			result = append(result, v)
		}
	}
	return result
}

// PlainRotateVec 实现明文向量的循环旋转，方向与Lattigo密文旋转一致（正数右移，负数左移）
// 输入: vec 原始向量，steps 旋转步长（正数右移，负数左移）
// 输出: 旋转后的新切片
func PlainRotateVec(vec []float64, steps int) []float64 {
	n := len(vec)
	if n == 0 {
		return nil
	}
	steps = ((steps % n) + n) % n // 保证步长在[0, n)范围
	res := make([]float64, n)
	for i := 0; i < n; i++ {
		res[(i+steps)%n] = vec[i]
	}
	return res
}

// ClassifierDense 计算 input * weight^T + bias^T（bias 按列广播）
// input: [m, n], weight: [k, n], bias: [k, 1] 或 [1, k]，输出 [m, k]
func ClassifierDense(input, weight, bias *mat.Dense) *mat.Dense {
	inRows, inCols := input.Dims()
	wRows, wCols := weight.Dims()
	bRows, bCols := bias.Dims()
	_ = inCols

	// 权重转置
	weightT := mat.NewDense(wCols, wRows, nil)
	weightT.CloneFrom(weight.T())

	// input * weight^T
	out := mat.NewDense(inRows, wRows, nil)
	out.Mul(input, weightT)

	// 偏置转置并按列广播加到每一行
	var biasT *mat.Dense
	if bCols == 1 {
		biasT = mat.NewDense(1, bRows, nil)
		biasT.CloneFrom(bias.T())
	} else {
		biasT = mat.NewDense(1, bCols, nil)
		biasT.CloneFrom(bias)
	}

	for i := 0; i < inRows; i++ {
		for j := 0; j < wRows; j++ {
			out.Set(i, j, out.At(i, j)+biasT.At(0, j))
		}
	}
	return out
}
