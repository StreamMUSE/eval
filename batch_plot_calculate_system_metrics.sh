experiment_name="experiments2-remote"

uv run src/plot/plot_system_performance.py ../StreamMUSE/${experiment_name}/realtime/ --plot-dir results-${experiment_name}/sys-plots
uv run compute_final_system_metric.py ../StreamMUSE/${experiment_name}/realtime/ -o results-${experiment_name}/final-sys-results