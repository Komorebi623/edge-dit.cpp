python3 scripts/run_parallel_collective_test.py \
  --binary build-parallel-cpu/bin/parallel-collective-test \
  --backend cpu \
  --world-size 7

python3 scripts/run_parallel_collective_test.py \
  --binary build-parallel-nccl/bin/parallel-collective-test \
  --backend nccl \
  --world-size 7