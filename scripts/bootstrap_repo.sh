#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d .git ]]; then
  git init
fi

git add .
git status --short

cat <<'EOF'

Next steps:
1. Create a GitHub repository.
2. Add it as a remote:
   git remote add origin <your-repo-url>
3. Commit and push:
   git commit -m "Prepare GPU cloud deployment"
   git branch -M main
   git push -u origin main
4. Watch GitHub Actions publish the GHCR image.
EOF
