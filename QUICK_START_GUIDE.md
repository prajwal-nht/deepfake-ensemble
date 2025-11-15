# 🚀 Quick Start Guide - Deepfake Detection

Since npm is not available in your system PATH, here's how to get everything running:

## Step 1: Start the Backend (Modal Client)

```bash
python start_backend_only.py
```

This will:
- Start the Modal Client backend on port 8000
- Show you all the test endpoints
- Handle the Modal API integration

## Step 2: Install Node.js (if not already installed)

1. Download Node.js from: https://nodejs.org/
2. Install it (this will also install npm)
3. Restart your terminal/PowerShell

## Step 3: Start the Frontend

### Option A: Using the batch file (Windows)
```bash
start_frontend.bat
```

### Option B: Manual commands
```bash
cd frontend
npm install
npm run dev
```

## 🧪 Testing the Backend

Once the backend is running, test these endpoints:

1. **Health Check**: http://localhost:8000/api/health
2. **Modal Connection Test**: http://localhost:8000/api/test-modal
3. **Face Indices**: http://localhost:8000/api/face_indices?user_id=test

Or run the test script:
```bash
python test_modal_client.py
```

## 🔗 Available Endpoints

### Backend (http://localhost:8000)
- `GET /api/health` - Health check
- `GET /api/test-modal` - Test Modal connection
- `POST /api/detect` - Deepfake detection
- `POST /api/face_indices` - Create face index
- `GET /api/face_indices?user_id={id}` - List face indices
- `DELETE /api/face_indices/{id}` - Delete face index

### Frontend (http://localhost:3000)
- `/` - Main detection interface
- `/face-index` - Face indexing (requires sign-in)

## 🔧 Troubleshooting

### Backend Issues
- **SSL Errors**: The backend now handles these automatically
- **Modal Connection**: Test with `/api/test-modal` endpoint
- **Port 8000 busy**: Kill any process using port 8000

### Frontend Issues
- **npm not found**: Install Node.js from nodejs.org
- **Port 3000 busy**: The frontend will suggest an alternative port

## 📝 What's Working Now

✅ Modal Client backend with SSL fixes  
✅ Face indexing endpoints  
✅ CORS configuration  
✅ Error handling and retry logic  
✅ Test endpoints for debugging  

## 🎯 Next Steps

1. Start the backend first
2. Test the Modal connection
3. Install Node.js if needed
4. Start the frontend
5. Test the full application

The backend should work immediately, and once you have Node.js installed, the frontend will connect seamlessly!