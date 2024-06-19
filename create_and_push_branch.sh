#!/bin/bash

# Function to handle errors and exit the script
handle_error() {
    echo "Error: $1"
    exit 1
}

# Clone the repository (if not already cloned)
if [ ! -d "school_budgets" ]; then
    git clone https://www.github.com/aurascoper/school_budgets.git || handle_error "Failed to clone repository"
fi

# Navigate to the repository
cd school_budgets || handle_error "Failed to navigate to repository"

# Create a new branch
branch_name="new-branch-name"
git checkout -b $branch_name || handle_error "Failed to create new branch"

# Add all files to the staging area
git add . || handle_error "Failed to stage changes"

# Commit the changes
git commit -m "Add initial files for new branch" || handle_error "Failed to commit changes"

# Push the new branch to the remote repository
git push origin $branch_name || handle_error "Failed to push new branch to remote repository"

echo "New branch '$branch_name' successfully created and pushed to remote repository"
