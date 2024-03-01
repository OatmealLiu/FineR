folder_path="scripts_eval"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sh "$file"
    fi
done