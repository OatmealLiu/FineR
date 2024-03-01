folder_path="scripts_eval_random"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sh "$file"
    fi
done