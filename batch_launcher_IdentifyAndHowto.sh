folder_path="scripts_IdentifyAndHowto"
for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        sh "$file"
    fi
done