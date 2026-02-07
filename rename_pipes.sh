find . -depth -exec bash -c '
for f; do
  new="$(printf "%s" "$f" | sed -e "s/|/_/g" -e "s/=//g" -e "s/,/_/g")"
  if [[ "$f" != "$new" ]]; then
    mv -- "$f" "$new"
  fi
done
' bash {} +
