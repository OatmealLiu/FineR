folder_path="scripts_full_pipeline"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done

folder_path="scripts_full_pipeline_random"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done