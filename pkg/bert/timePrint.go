package bert

// ===== stats.go (可直接放在同目录内) =====
import (
	"fmt"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

/* ──────────────── 数据结构 ──────────────── */

type opStat struct {
	total time.Duration
	count int
}

type profiler struct {
	mu        sync.Mutex
	opStats   map[string]*opStat
	peakAlloc uint64
	prefix    string
}

func newProfiler() *profiler { return &profiler{opStats: make(map[string]*opStat)} }

func (p *profiler) WithPrefix(pre string) *profiler {
	if pre == "" {
		return p
	}
	return &profiler{opStats: p.opStats, prefix: p.prefix + pre}
}

/* ──────────────── 计时入口 ──────────────── */

func (p *profiler) Enter(tag string) func() {
	full := p.prefix + tag
	start := time.Now()
	return func() {
		dur := time.Since(start)

		p.mu.Lock()
		s := p.opStats[full]
		if s == nil {
			s = &opStat{}
			p.opStats[full] = s
		}
		s.total += dur
		s.count++

		// 峰值内存
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		if m.Alloc > p.peakAlloc {
			p.peakAlloc = m.Alloc
		}
		p.mu.Unlock()
	}
}

/* ──────────────── 报表 ──────────────── */

func (p *profiler) Report(batch int) {
	/* 列宽设置 */
	const (
		compW  = 24
		callW  = 8
		timeW  = 10
		avgW   = 12
		ratioW = 10
	)
	rowW := 1 + compW + 2 + callW + 2 + timeW + 2 + avgW + 2 + ratioW + 1
	hLine := func(l, r string) string { return l + strings.Repeat("─", rowW-2) + r }

	/* 顶层固定顺序 */
	top := []string{
		"Attention/Pt-ct MatMul", "Attention/MultiHead", "Attention/Bootstrap-1",
		"Self/Pt-ct MatMul", "Self/LayerNorm", "Self/Bootstrap-2",
		"Inter/Pt-ct-1 MatMul", "Inter/GELU", "Inter/Pt-ct-2 MatMul", "Inter/Bootstrap-3",
		"Final/LayerNorm", "Final/Bootstrap-4",
	}
	sepAfter := map[int]struct{}{2: {}, 5: {}, 9: {}}

	/* 总时长 */
	var grand time.Duration
	for _, v := range p.opStats {
		grand += v.total
	}
	grandSec := grand.Seconds()

	pct := func(x float64) string {
		if x < 0.01 {
			return fmt.Sprintf("%*.3f", ratioW, x)
		}
		return fmt.Sprintf("%*.2f", ratioW, x)
	}

	/* ────── 打印开始 ────── */
	fmt.Println()
	fmt.Println(hLine("┌", "┐"))
	title := "  Breakdown of Encrypted BERT-Tiny Inference (batch=" + strconv.Itoa(batch) + ")"
	fmt.Printf("│%-*s│\n", rowW-2, title)

	seg := func() {
		fmt.Println("├" +
			strings.Repeat("─", compW+2) + "┼" +
			strings.Repeat("─", callW+2) + "┼" +
			strings.Repeat("─", timeW+2) + "┼" +
			strings.Repeat("─", avgW+2) + "┼" +
			strings.Repeat("─", ratioW+2) + "┤")
	}
	seg()
	fmt.Printf("│ %-*s │ %*s │ %*s │ %*s │ %*s │\n",
		compW, "Component · Operation",
		callW, "Calls",
		timeW, "Time/s",
		avgW, "Avg/batch s",
		ratioW, "Avg/total %")
	seg()

	/* ------ 辅助：打印一行 ------ */
	printRow := func(name string, s *opStat) {
		if s == nil {
			fmt.Printf("│ %-*s │ %*s │ %*s │ %*s │ %*s │\n",
				compW, name, callW, "-", timeW, "-", avgW, "-", ratioW, "-")
			return
		}
		sec := s.total.Seconds()
		fmt.Printf("│ %-*s │ %*d │ %*.2f │ %*.4f │ %s │\n",
			compW, name,
			callW, s.count,
			timeW, sec,
			avgW, sec/float64(batch),
			pct(100*sec/grandSec))
	}

	/* ------ 打印顶层及其子项 ------ */
	seen := make(map[string]struct{})

	for idx, t := range top {
		printRow(t, p.opStats[t])
		seen[t] = struct{}{}

		// 打印所有以 t+"/" 开头的子项
		prefix := t + "/"
		var children []string
		for k := range p.opStats {
			if strings.HasPrefix(k, prefix) {
				children = append(children, k)
			}
		}
		sort.Strings(children)
		for _, c := range children {
			seen[c] = struct{}{}
			indentName := "  " + c[len(prefix):] // 两空格缩进
			printRow(indentName, p.opStats[c])
		}

		if _, ok := sepAfter[idx]; ok {
			seg()
		}
	}

	/* ------ 打印未列入 top 的散项 ------ */
	var leftovers []string
	for k := range p.opStats {
		if _, ok := seen[k]; !ok {
			leftovers = append(leftovers, k)
		}
	}
	if len(leftovers) > 0 {
		seg()
		sort.Strings(leftovers)
		for _, k := range leftovers {
			printRow(k, p.opStats[k])
		}
	}

	/* Total */
	seg()
	fmt.Printf("│ %-*s │ %*s │ %*.2f │ %*.4f │ %*.2f │\n",
		compW, "Total",
		callW, "—",
		timeW, grandSec,
		avgW, grandSec/float64(batch),
		ratioW, 100.0)
	fmt.Println(hLine("└", "┘"))

	fmt.Printf("Peak HeapAlloc: %.2f GiB\n", float64(p.peakAlloc)/(1<<30))
}
