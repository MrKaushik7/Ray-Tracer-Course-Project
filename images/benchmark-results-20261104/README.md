# Benchmark plots

Source: `/Users/adithyakaushik/Desktop/IITM/Sem7/CS6023/ray-tracer-cs23b002/results/20261104-182900-395464`.

Normal timing plots use means and sample standard deviations of three process-level CUDA-event averages (20 timed launches per process). Error bars are not confidence intervals. Scene 5001 has a different arrangement and is excluded from scaling curves. Ratio charts use ratios of means without propagated uncertainty. Profiler timings are excluded. No image-quality equivalence is implied by speedups.

- [01-normal-scaling.png](01-normal-scaling.png)
- [02-normal-5000-all-variants.png](02-normal-5000-all-variants.png)
- [03-normal-bvh-speedups.png](03-normal-bvh-speedups.png)
- [04-fast-math-speedup-heatmap.png](04-fast-math-speedup-heatmap.png)
- [05-max-opt-normal-vs-fast-math.png](05-max-opt-normal-vs-fast-math.png)
- [06-bvh-scene-5000-vs-5001.png](06-bvh-scene-5000-vs-5001.png)

Regenerate with `python3 scripts/plot_benchmark_results.py` (requires gnuplot with pngcairo).
