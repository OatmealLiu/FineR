folder_path="scripts_eval"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done

folder_path="scripts_eval_random"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sbatch "$file"
    fi
done