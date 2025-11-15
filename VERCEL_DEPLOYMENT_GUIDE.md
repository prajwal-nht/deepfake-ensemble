# Vercel Deployment Guide

## Prerequisites
1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **GitHub Repository**: Your code should be in a GitHub repository
3. **Modal Backend**: Your Modal backend should be deployed and accessible

## Step 1: Get Your Modal Backend URL

First, you need to get your Modal backend URL. Run your Modal app and note the URL:

```bash
cd Deepfake-API/Deepfake/deepfake-detection
modal serve modal_client.py
```

The URL will look like: `https://your-username--modal-client-app.modal.run`

## Step 2: Update Environment Variables

Update the following files with your actual Modal URL:

### frontend/.env.production
```env
VITE_CLERK_PUBLISHABLE_KEY=pk_test_bmVhcmJ5LXRlcm1pdGUtNjUuY2xlcmsuYWNjb3VudHMuZGV2JA
VITE_API_BASE_URL=https://your-actual-modal-url.modal.run
```

### frontend/vercel.json
Update the `VITE_API_BASE_URL` in the build env section with your actual Modal URL.

## Step 3: Update Backend CORS Settings

Your Modal backend needs to allow requests from your Vercel domain. Update the CORS middleware in `modal_client.py`:

```python
# In the add_cors_headers middleware function
response.headers["Access-Control-Allow-Origin"] = "https://your-vercel-app.vercel.app"
```

Or allow multiple origins:
```python
allowed_origins = [
    "http://localhost:3000",  # Development
    "https://your-vercel-app.vercel.app"  # Production
]
origin = request.headers.get("origin")
if origin in allowed_origins:
    response.headers["Access-Control-Allow-Origin"] = origin
```

## Step 4: Deploy to Vercel

### Option A: Vercel CLI (Recommended)
1. Install Vercel CLI:
   ```bash
   npm i -g vercel
   ```

2. Navigate to your frontend directory:
   ```bash
   cd frontend
   ```

3. Login to Vercel:
   ```bash
   vercel login
   ```

4. Deploy:
   ```bash
   vercel --prod
   ```

### Option B: Vercel Dashboard
1. Go to [vercel.com/dashboard](https://vercel.com/dashboard)
2. Click "New Project"
3. Import your GitHub repository
4. Set the root directory to `frontend`
5. Vercel will auto-detect it's a Vite project
6. Add environment variables:
   - `VITE_CLERK_PUBLISHABLE_KEY`: Your Clerk key
   - `VITE_API_BASE_URL`: Your Modal backend URL
7. Click "Deploy"

## Step 5: Update Backend CORS with Your Vercel URL

Once deployed, you'll get a Vercel URL like `https://your-app-name.vercel.app`. Update your Modal backend CORS settings with this URL.

## Step 6: Test Your Deployment

1. Visit your Vercel URL
2. Test the deepfake detection functionality
3. Check browser console for any CORS or API errors

## Troubleshooting

### CORS Issues
- Ensure your Modal backend allows your Vercel domain
- Check browser console for specific CORS errors
- Verify the API URL is correct in your environment variables

### Build Issues
- Check Vercel build logs in the dashboard
- Ensure all dependencies are in package.json
- Verify TypeScript compilation passes locally

### API Connection Issues
- Verify your Modal backend is running and accessible
- Check the API URL format (should include https://)
- Test API endpoints directly in browser or Postman

## Environment Variables Summary

For production deployment, you need:
- `VITE_CLERK_PUBLISHABLE_KEY`: Your Clerk authentication key
- `VITE_API_BASE_URL`: Your Modal backend URL (e.g., https://username--app-name.modal.run)

## Next Steps

After successful deployment:
1. Set up custom domain (optional)
2. Configure production analytics
3. Set up monitoring and error tracking
4. Consider implementing caching strategies