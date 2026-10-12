Outside tolerance, about 3.6× slower than predicted, as the tolerance note
anticipated. The dominant term is right: the GPU is idle most of the time.
The cost per op is wrong.

From [profiler.txt](profiler.txt), one profiled pass of the loop:

- 10,000 `aten::add_` calls launched exactly 10,000 kernels, and the GPU ran
  each one for 1.15 μs on average: 11.5 ms of GPU time in total.
- Each `add_` cost 7.1 μs of host time, of which `cudaLaunchKernel` took
  3.9 μs on average. The profiler adds overhead: the profiled pass took 71 ms,
  the timed passes 61.6 ms.

The unprofiled loop costs 6.16 μs per op, against napkin's 1.69 μs launch
constant. That constant is a back-to-back `cuLaunchKernel` through the driver
API from Rust (napkin-bench). It does not include the CUDA runtime wrapper, the
PyTorch dispatcher or the Python interpreter, which together cost about
4.5 μs per op on this host.

For an eager-mode framework loop, treat `--launches` × the launch constant as
a lower bound. Measure the framework's own per-op cost, about 6 μs here, before
sizing a launch-bound job. CUDA graphs or fusion remove the term, as the case
excludes.
