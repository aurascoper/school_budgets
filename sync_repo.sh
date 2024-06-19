#!/bin/bash

# fxn to handle errors and exit the scirpt
handle_error() {
	echo "Error: $1"
	exit 1
}

# pull the latest changes from the remote repository
git pull origin main --allow-unrelated-histories || handle_error "failed to pull latest changes from remote repository"

# check for merge conflicts
if grep -q '<<<<<<<' README.md; then
	echo "Merge conflict detected in README.md. Please resolve manually."
	exit 1
fi

# stage all changes
git add . || handle_error "Failed to stage changes"

# commit the changes
git commit -m "Sync local changes with remote repository" || handle_error "Failed to commit changes"

# push changes to remote repository
 git push origin main || handle_error "Failed to push changes to remote repo"

echo "Local repository successfully synced with remote repository. Bon s'affairres"
