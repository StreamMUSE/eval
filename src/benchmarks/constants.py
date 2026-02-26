"""Standard paths and settings for benchmark analysis."""

# The three real-time deployment settings used in the paper.
REAL_TIME_SETTINGS = ("remote", "local", "local_server")

# Default subdir names under StreamMUSE bench_results for each setting.
# Users can place CSVs in bench_results/remote/, bench_results/local/, bench_results/local_server/.
# Legacy: manual_remote, manual_local, manual_local_server are also accepted as input dir names;
# the analyzer accepts any dir path and --setting is used for the output subdir name.
BENCH_RESULTS_SUBDIRS = {
    "remote": "remote",
    "local": "local",
    "local_server": "local_server",
}

# Output subdirs under eval results/benchmarks/
BENCHMARK_OUTPUT_SUBDIRS = list(BENCH_RESULTS_SUBDIRS.keys())
