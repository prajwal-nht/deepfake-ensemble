@echo off
echo ========================================
echo Pushing to GitHub: deepfake-ensemble
echo ========================================
echo.

REM Go to root directory
cd /d D:\deepfake

echo Step 1: Checking git status...
git status
echo.

echo Step 2: Adding all files...
git add .
echo.

echo Step 3: Committing changes...
git commit -m "Initial commit: Ensemble deepfake detection system with 5-model architecture and demographic weighting"
echo.

echo Step 4: Verifying remote...
git remote -v
echo.

echo Step 5: Creating main branch and pushing...
git branch -M main
git push -u origin main
echo.

echo ========================================
echo Done! Check https://github.com/prajwal-nht/deepfake-ensemble
echo ========================================
pause
