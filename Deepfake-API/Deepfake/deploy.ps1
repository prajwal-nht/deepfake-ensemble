# Check disk space
$disk = Get-PSDrive -PSProvider 'FileSystem' | Where-Object { $_.Root -eq (Get-Location).Drive.Root }
$freeSpaceGB = [math]::Round($disk.Free / 1GB, 2)

if ($freeSpaceGB -lt 5) {
    Write-Host "❌ Error: Not enough disk space. Need at least 5GB free, but only $freeSpaceGB GB available."
    exit 1
}

Write-Host "✅ Available disk space: $freeSpaceGB GB"

# Create a temporary directory for deployment
$tempDir = "$env:TEMP\deepfake-deploy-$(Get-Date -Format 'yyyyMMddHHmmss')" 
New-Item -ItemType Directory -Path $tempDir -Force

# Define files and directories to exclude (large or unnecessary files)
$excludeDirs = @(
    "__pycache__",
    ".git",
    ".github",
    "venv*",
    "env*",
    ".venv*",
    "*.egg-info",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".pytest_cache",
    "*.so",
    "*.dll",
    "*.dylib",
    "*.a",
    "*.lib",
    "*.h5",
    "*.hdf5",
    "*.pt",
    "*.pth",
    "*.bin",
    "*.onnx",
    "*.tflite",
    "*.pb",
    "*.ckpt"
)

# Install required Python packages
Write-Host "Installing required Python packages..."
pip install -r requirements.txt

# Install Modal if not already installed
if (-not (Get-Command modal -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Modal CLI..."
    pip install modal
}

# Deploy to Modal
Write-Host "Deploying to Modal..."
modal deploy modal_deploy_clean.py

# Get the deployment URL
Write-Host "Getting deployment URL..."
$deployment_url = modal app list | Select-String -Pattern "deepfake-detection-api" -Context 0,1 | 
    Select-String -Pattern "https://" | 
    ForEach-Object { $_.Matches[0].Value }

Write-Host "`nDeployment successful!"
Write-Host "API URL: $deployment_url"
Write-Host "`nTest the API with:"

# Create a simple main.py for Modal
@'
from modal_deploy_final import web_app

app = web_app
'@ | Out-File -FilePath "$tempDir\main.py" -Encoding utf8

# Create a .dockerignore file to exclude large files
@'
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
.env
.idea/
.vscode/
*.egg-info/
.eggs/
dist/
build/
*.so
*.dll
*.dylib
*.a
*.lib
*.h5
*.hdf5
*.pt
*.pth
*.bin
*.onnx
*.tflite
*.pb
*.ckpt
'@ | Out-File -FilePath "$tempDir\.dockerignore" -Encoding utf8

# Deploy using Modal
Write-Host "🚀 Starting deployment..."
try {
    # Deploy using Modal
    Set-Location $tempDir
    modal deploy deploy_clean.py
} catch {
    Write-Host "❌ Deployment failed: $_"
} finally {
    # Clean up
    Set-Location "$PSScriptRoot"
    try {
        Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
    } catch {
        Write-Host "⚠️ Warning: Could not clean up temporary directory: $_"
    }
}

Write-Host "✨ Deployment process completed"
