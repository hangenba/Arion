package plain

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

// ----------------------- Attention primitives ----------------------- //

func softmaxVec(vec *mat.VecDense) *mat.VecDense {
	n := vec.Len()
	max := vec.AtVec(0)
	// 求最大值
	for i := 1; i < n; i++ {
		if val := vec.AtVec(i); val > max {
			max = val
		}
	}
	sum := 0.0
	for i := 0; i < n; i++ {
		sum += math.Exp(vec.AtVec(i) - max)
	}
	out := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		out.SetVec(i, math.Exp(vec.AtVec(i)-max)/sum)
	}
	return out
}

func softmaxRows(m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	out := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		row := m.RowView(i).(*mat.VecDense)
		sm := softmaxVec(row)
		for j := 0; j < c; j++ {
			out.Set(i, j, sm.AtVec(j))
		}
	}
	return out
}

func scaledDotProductAttention(q, k, v *mat.Dense) *mat.Dense {
	_, dk := k.Dims()
	kT := mat.DenseCopyOf(k.T())
	var scores mat.Dense
	scores.Mul(q, kT)
	scale := 1.0 / math.Sqrt(float64(dk))
	scores.Scale(scale, &scores)
	weights := softmaxRows(&scores)
	var output mat.Dense
	output.Mul(weights, v)
	return &output
}

func splitHeads(x *mat.Dense, numHeads int) []*mat.Dense {
	r, c := x.Dims()
	if c%numHeads != 0 {
		log.Fatalf("hidden size %d not divisible by %d", c, numHeads)
	}
	dHead := c / numHeads
	heads := make([]*mat.Dense, numHeads)
	for h := 0; h < numHeads; h++ {
		block := mat.NewDense(r, dHead, nil)
		for i := 0; i < r; i++ {
			for j := 0; j < dHead; j++ {
				block.Set(i, j, x.At(i, h*dHead+j))
			}
		}
		heads[h] = block
	}
	return heads
}

func concatHeads(heads []*mat.Dense) *mat.Dense {
	if len(heads) == 0 {
		return nil
	}
	r, dHead := heads[0].Dims()
	numHeads := len(heads)
	out := mat.NewDense(r, numHeads*dHead, nil)
	for i := 0; i < r; i++ {
		for h := 0; h < numHeads; h++ {
			for j := 0; j < dHead; j++ {
				out.Set(i, h*dHead+j, heads[h].At(i, j))
			}
		}
	}
	return out
}

func LinearTransform(x, weight *mat.Dense, bias *mat.VecDense, realRows int) *mat.Dense {
	batch, inDim := x.Dims()
	wRows, outDim := weight.Dims()
	if inDim != wRows || bias.Len() != outDim {
		log.Fatalf("LinearTransform: 维度不匹配 x:(%d,%d) weight:(%d,%d) bias:(%d)", batch, inDim, wRows, outDim, bias.Len())
	}
	// x * weight
	var out mat.Dense
	out.Mul(x, weight) // [batch, out_dim]

	// 仅对前 realRows 行加 bias
	if realRows > batch {
		log.Fatalf("realRows (%d) 超过输入行数 (%d)", realRows, batch)
	}
	for i := 0; i < realRows; i++ {
		for j := 0; j < outDim; j++ {
			out.Set(i, j, out.At(i, j)+bias.AtVec(j))
		}
	}
	return &out
}

func mulArionadAttention(x *mat.Dense, numHeads int) *mat.Dense {
	qHeads := splitHeads(x, numHeads)
	kHeads := splitHeads(x, numHeads)
	vHeads := splitHeads(x, numHeads)
	headsOut := make([]*mat.Dense, numHeads)
	for h := 0; h < numHeads; h++ {
		headsOut[h] = scaledDotProductAttention(qHeads[h], kHeads[h], vHeads[h])
	}
	return concatHeads(headsOut)
}

// --------------------------- I/O helpers --------------------------- //

func loadCSVFile(row, col int, path string) *mat.Dense {
	f, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	rdr := csv.NewReader(f)
	records, err := rdr.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	data := make([]float64, 0, row*col)
	for i := 0; i < row && i < len(records); i++ {
		record := records[i]
		for j := 0; j < col && j < len(record); j++ {
			v, err := strconv.ParseFloat(record[j], 64)
			if err != nil {
				log.Fatal(err)
			}
			data = append(data, v)
		}
		// 若该行不足col，则补0
		for j := len(record); j < col; j++ {
			data = append(data, 0)
		}
	}
	// 若总行数不足row，则补0行
	for i := len(records); i < row; i++ {
		for j := 0; j < col; j++ {
			data = append(data, 0)
		}
	}
	return mat.NewDense(row, col, data)
}

func loadCSVVec(path string) *mat.VecDense {
	f, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	rdr := csv.NewReader(f)
	records, err := rdr.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	data := make([]float64, 0, len(records))
	for _, record := range records {
		// 只取每行的第一个元素
		if len(record) > 0 {
			v, err := strconv.ParseFloat(record[0], 64)
			if err != nil {
				log.Fatal(err)
			}
			data = append(data, v)
		}
	}
	return mat.NewVecDense(len(data), data)
}

func saveCSV(path string, m *mat.Dense) {
	f, err := os.Create(path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	rows, cols := m.Dims()
	for i := 0; i < rows; i++ {
		row := make([]string, cols)
		for j := 0; j < cols; j++ {
			row[j] = fmt.Sprintf("%f", m.At(i, j))
		}
		if err := w.Write(row); err != nil {
			log.Fatal(err)
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		log.Fatal(err)
	}
}

func transposeDense(src *mat.Dense) *mat.Dense {
	r, c := src.Dims()
	dst := mat.NewDense(c, r, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			dst.Set(j, i, src.At(i, j))
		}
	}
	return dst
}

func MatricesMaxMin(mats []*mat.Dense) (maxVal, minVal float64) {
	maxVal = math.SmallestNonzeroFloat64
	minVal = math.MaxFloat64
	for _, m := range mats {
		r, c := m.Dims()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				v := m.At(i, j)
				if v > maxVal {
					maxVal = v
				}
				if v < minVal {
					minVal = v
				}
			}
		}
	}
	return
}
