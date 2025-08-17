# build programs
clang -Wall -Wextra -Wpedantic -O3 -funroll-loops -march=native -o ./bin/2_mvp 2_mvp.c
clang -Wall -Wextra -Wpedantic -O3 -funroll-loops -march=native -o ./bin/3_simd 3_simd.c
clang -Wall -Wextra -Wpedantic -O3 -funroll-loops -march=native -pthread -o ./bin/4_threads 4_threads.c
clang -Wall -Wextra -Wpedantic -O3 -funroll-loops -march=native -pthread -o ./bin/6_mmap 6_mmap.c

# build benchmark gen
clang -Wall -Wextra -Wpedantic -O3 -funroll-loops -march=native -o ./bin/setup_benchmark setup_benchmark.c

# generate benchmark
rm -f bench.txt
./bin/setup_benchmark

# assert C versions match 65076996
./bin/2_mvp bench.txt | grep -q 65076996 || echo "2_mvp failed" && exit 1
./bin/3_simd bench.txt | grep -q 65076996 || echo "3_simd failed" && exit 1
./bin/4_threads bench.txt | grep -q 65076996 || echo "4_threads failed" && exit 1
./bin/6_mmap bench.txt | grep -q 65076996 || echo "6_mmap failed" && exit 1

# run benchmark
hyperfine --warmup 2 --min-runs 3 \
    'python3 0_mvp.py bench.txt' \
    'python3 1_c_regex.py bench.txt' \
    './bin/2_mvp bench.txt' \
    './bin/3_simd bench.txt' \
    './bin/4_threads bench.txt' \
    './bin/6_mmap bench.txt'
