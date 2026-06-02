python3 scripts/run_parallel_collective_test.py \
  --binary ./build-test/bin/parallel-collective-test \
  --backend cpu \
  --world-size 7 \
  --keep-store

python3 scripts/run_parallel_collective_test.py \
  --binary build-parallel-nccl/bin/parallel-collective-test \
  --backend nccl \
  --world-size 7