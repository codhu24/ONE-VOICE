# Git Setup and Push Guide for OneVoice

## ⚠️ IMPORTANT: Before You Push

**CRITICAL**: Your `.env` file contains sensitive API keys that are currently visible in this guide. These keys MUST NOT be pushed to Git!

### Current API Keys Found:
- OpenAI API Key
- Google Translate API Key
- Google API Key

**Action Required**: After pushing to Git, you should:
1. Rotate (regenerate) all API keys from their respective platforms
2. Update your local `.env` file with new keys
3. Never share these keys publicly

---

## Step-by-Step Git Push Guide

### 1. Verify .gitignore is Working

First, check what files will be committed:

```bash
git status
```

**Verify that these files are NOT listed:**
- `backend/.env`
- `backend/__pycache__/`
- `frontend/node_modules/`
- `backend/.venv/` or `backend/venv/`
- SSH keys (`1`, `1.pub`)

### 2. Remove Sensitive Files from Git History (If Already Tracked)

If `.env` or SSH keys were previously committed, remove them:

```bash
# Remove .env from Git tracking
git rm --cached backend/.env

# Remove SSH keys from Git tracking
git rm --cached 1
git rm --cached 1.pub

# Remove __pycache__ if tracked
git rm -r --cached backend/__pycache__
```

### 3. Add Files to Staging

Add all files that should be committed:

```bash
# Add all files (respecting .gitignore)
git add .

# Or add specific files
git add README.md
git add UI_UX_DESIGN.md
git add backend/main.py
git add backend/requirements.txt
git add frontend/
```

### 4. Verify What Will Be Committed

```bash
git status
```

**Double-check that sensitive files are NOT in the staging area!**

### 5. Commit Your Changes

```bash
git commit -m "Update project structure and documentation

- Updated UI/UX design documentation
- Enhanced backend API endpoints
- Added multilingual support with LanguageContext
- Integrated WebSocket voice service
- Updated README with current project structure"
```

### 6. Push to Remote Repository

#### If you already have a remote repository:

```bash
# Push to main/master branch
git push origin master

# Or if your branch is named 'main'
git push origin main
```

#### If you DON'T have a remote repository yet:

**Option A: Create a new repository on GitHub**

1. Go to [GitHub](https://github.com) and create a new repository
2. Don't initialize with README (you already have one)
3. Copy the repository URL
4. Add the remote and push:

```bash
# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to remote
git push -u origin master
```

**Option B: Create a new repository on GitLab**

1. Go to [GitLab](https://gitlab.com) and create a new project
2. Copy the repository URL
3. Add the remote and push:

```bash
# Add remote repository
git remote add origin https://gitlab.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to remote
git push -u origin master
```

---

## Verify Your Push

After pushing, check your repository online to ensure:

✅ All code files are present
✅ `.env` file is NOT visible
✅ SSH keys (`1`, `1.pub`) are NOT visible
✅ `node_modules/` is NOT present
✅ `__pycache__/` is NOT present
✅ `.env.example` IS present (for others to know what variables are needed)

---

## Setting Up for Collaborators

### Create .env.example Files

For backend, create a template:

**backend/.env.example:**
```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_TRANSLATE_API_KEY=your_google_translate_key_here
GOOGLE_API_KEY=your_google_api_key_here
HOST=127.0.0.1
PORT=8000
```

For frontend (if needed):

**frontend/.env.example:**
```env
VITE_API_BASE=http://127.0.0.1:8000
```

Then add these example files:

```bash
git add backend/.env.example
git add frontend/.env.example
git commit -m "Add environment variable templates"
git push
```

---

## Common Git Commands Reference

### Check Status
```bash
git status                    # See what's changed
git log --oneline            # View commit history
git diff                     # See unstaged changes
```

### Branching
```bash
git branch feature-name      # Create new branch
git checkout feature-name    # Switch to branch
git checkout -b feature-name # Create and switch
git merge feature-name       # Merge branch into current
```

### Undoing Changes
```bash
git restore file.txt         # Discard changes to file
git restore --staged file.txt # Unstage file
git reset HEAD~1             # Undo last commit (keep changes)
git reset --hard HEAD~1      # Undo last commit (discard changes)
```

### Remote Operations
```bash
git remote -v                # View remote repositories
git fetch origin             # Fetch changes from remote
git pull origin master       # Pull and merge changes
git push origin master       # Push commits to remote
```

---

## Security Best Practices

### 1. Never Commit Secrets
- API keys
- Passwords
- Private keys
- Database credentials
- OAuth tokens

### 2. Use Environment Variables
- Store secrets in `.env` files
- Add `.env` to `.gitignore`
- Provide `.env.example` templates

### 3. Rotate Compromised Keys
If you accidentally commit secrets:
1. Remove them from Git history (use `git filter-branch` or BFG Repo-Cleaner)
2. Rotate all exposed keys immediately
3. Force push to remote: `git push --force`

### 4. Use .gitignore Properly
- Add sensitive patterns before first commit
- Review `.gitignore` regularly
- Test with `git status` before committing

---

## Troubleshooting

### Problem: ".env file is showing in git status"

**Solution:**
```bash
git rm --cached backend/.env
git commit -m "Remove .env from tracking"
```

### Problem: "Large files causing push to fail"

**Solution:**
```bash
# Check file sizes
git ls-files -s | awk '{print $4, $2}' | sort -n -r | head -20

# Use Git LFS for large files
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

### Problem: "Merge conflicts"

**Solution:**
```bash
# Pull latest changes
git pull origin master

# Resolve conflicts in files
# Edit conflicted files manually

# Mark as resolved
git add conflicted-file.txt
git commit -m "Resolve merge conflicts"
git push
```

---

## Quick Command Checklist

```bash
# 1. Check status
git status

# 2. Add files
git add .

# 3. Commit
git commit -m "Your message here"

# 4. Push
git push origin master
```

---

## Additional Resources

- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [GitLab Documentation](https://docs.gitlab.com/)
- [Atlassian Git Tutorials](https://www.atlassian.com/git/tutorials)

---

**Last Updated**: November 2024
**Project**: OneVoice - Assistive Communication Platform
