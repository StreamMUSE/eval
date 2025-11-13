results_folder="results-experiment2-remote"

uv run src/plot/plot_jsd.py ${results_folder}/*.json -o ${results_folder}/jsd_all_models.png  
uv run src/plot/plot_inter_track_continuity.py ${results_folder}/*.json -o ${results_folder}/itc_all_models.png
uv run src/plot/plot_harmonicity_simple.py ${results_folder}/*.json -o ${results_folder}/harmonicity_all_models.png
uv run src/plot/plot_jsd.py ${results_folder}/*.json -o ${results_folder}/jsd_all_models_violin.png --plot-type violin
uv run src/plot/plot_inter_track_continuity.py ${results_folder}/*.json -o ${results_folder}/itc_all_models_violin.png --plot-type violin
uv run src/plot/plot_harmonicity_simple.py ${results_folder}/*.json -o ${results_folder}/harmonicity_all_models_violin.png --plot-type violin