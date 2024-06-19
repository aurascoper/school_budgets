#!/bin/bash

# Function to handle errors and exit the script
handle_error() {
    echo "Error: $1"
    exit 1
}

# Navigate to the repository
cd ~/Projects/EngineeringDesign/school_budgets || handle_error "Failed to navigate to repository"

# Pull the latest changes from the remote repository
git pull origin main --allow-unrelated-histories || handle_error "Failed to pull latest changes from remote repository"

# Check for merge conflicts
if grep -q '<<<<<<<' README.md; then
    echo "Merge conflict detected in README.md. Please resolve manually."
    exit 1
fi

# Add all files to the staging area
git add . || handle_error "Failed to stage changes"

# Commit the changes
git commit -m "Sync local changes with remote repository" || handle_error "Failed to commit changes"

# Push changes to remote repository
git push origin main || handle_error "Failed to push changes to remote repository"

echo "Local repository successfully synced with remote repository. Bon s'affairres"

# Optional: Automate merging process if needed
while true; do
    git pull origin main --allow-unrelated-histories || handle_error "Failed to pull latest changes from remote repository"
    git add . || handle_error "Failed to stage changes"
    git commit -m "Sync local changes with remote repository" || handle_error "Failed to commit changes"
    git push origin main || handle_error "Failed to push changes to remote repository"
    echo "Attempted to resolve and push changes. Checking for conflicts..."
    
    if grep -q '<<<<<<<' README.md; then
        echo "Merge conflict detected in README.md. Please resolve manually."
        exit 1
    else
        echo "Merge successful and repository is up-to-date. Bon s'affairres"
        break
    fi
done
