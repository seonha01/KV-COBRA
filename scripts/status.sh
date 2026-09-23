#!/usr/bin/env bash
# Progress of the reproduction queue: finished cells per result file + queue log tail.
cd "$(dirname "$0")/.."
echo "== result files (rows incl. FP16) =="
for k in ppl zeroshot longbench; do
  for f in results/$k/*.csv; do
    [ -f "$f" ] || continue
    n=$(( $(wc -l < "$f") - 1 ))
    printf "  %-10s %-24s %4d rows\n" "$k" "$(basename "$f" .csv)" "$n"
  done
done
echo "== queue log (last 8) =="
tail -n 8 results/logs/queue.log 2>/dev/null
echo "== GPUs =="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
