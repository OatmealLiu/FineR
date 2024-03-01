folder_path="scripts_full_pipeline"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sh "$file"
    fi
done