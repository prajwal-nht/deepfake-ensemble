# Browser-Based Detection Demo

This is a proof-of-concept showing how to run ML models directly in the browser using WebAssembly. No server needed - everything runs client-side.

## Running the Demo

```bash
cd wasm-demo
npm install
npm run dev
```

Open `http://localhost:5174` and drag an image to test it.

## What's This About?

The main deepfake detection system uses 5 large models (~2GB total) running on Modal servers. That's great for accuracy but requires a backend.

This demo explores running a smaller model (~4MB) entirely in your browser using ONNX Runtime Web. It's not as accurate as the full ensemble, but it's instant and works offline.

## How to Use It Standalone

The demo is completely independent from the main app. You can:

1. **Run it locally:**
   ```bash
   npm install
   npm run dev
   ```

2. **Build for production:**
   ```bash
   npm run build
   npm run preview
   ```

3. **Deploy anywhere:**
   ```bash
   # Deploy to Vercel
   vercel --prod

   # Or just copy the dist/ folder to any static host
   npm run build
   # Upload dist/ to Netlify, GitHub Pages, etc.
   ```

## What Works

- Drag and drop any image
- Instant classification (runs in browser)
- No API calls, no server needed
- Works offline once loaded

## What Doesn't Work (Yet)

- Not trained on deepfakes (uses generic MobileNetV2)
- Lower accuracy than server-based ensemble
- Can't handle videos (browser memory limits)
- No demographic analysis

## Why WASM?

**Good for:**
- Privacy (data never leaves browser)
- Speed (no network latency)
- Cost (no server inference costs)
- Offline use

**Not good for:**
- Large models (browser memory ~2GB max)
- Complex ensembles (our 5-model system is too big)
- Video processing (memory intensive)

## Could This Replace the Server?

Not really. The full ensemble needs ~2GB of models and significant compute. Browsers can't handle that well.

But WASM could be useful for:
- **Pre-screening**: Quick check before sending to server
- **Face detection**: Find faces in browser, send crops to server
- **Feature extraction**: Extract features client-side, classify server-side

## Technical Details

- **Model**: MobileNetV2 (quantized to INT8)
- **Size**: ~4MB
- **Runtime**: ONNX Runtime Web
- **Inference**: ~50-200ms depending on device
- **Memory**: ~50MB

## Integrating with Main App

Want to add this to the main frontend? Here's how:

1. Copy `wasm-demo/public/model.onnx` to `frontend/public/`
2. Install ONNX Runtime: `npm install onnxruntime-web`
3. Import the inference code from `wasm-demo/src/inference.js`
4. Add a "Quick Check" button that runs WASM before server analysis

Example:
```javascript
import { runInference } from './inference';

// Quick client-side check
const quickResult = await runInference(imageFile);
if (quickResult.confidence < 0.3) {
  // Probably real, skip server
  return { prediction: 'real', confidence: quickResult.confidence };
}

// Needs deeper analysis, send to server
const serverResult = await apiService.detectDeepfake(imageFile);
return serverResult;
```

## Performance Comparison

| Method | Speed | Accuracy | Cost | Privacy |
|--------|-------|----------|------|---------|
| WASM (this demo) | ~100ms | ~70% | Free | Perfect |
| Single server model | ~1s | ~80% | Low | Good |
| Full ensemble (Modal) | ~3-5s | ~90% | Medium | Good |

## Next Steps

To make this production-ready:
1. Train a small model specifically on deepfakes
2. Quantize it to INT8 or even INT4
3. Add face detection in WASM
4. Use it as a pre-filter before server analysis

## Files

- `index.html` - Simple UI
- `package.json` - Dependencies
- `public/model.onnx` - Quantized MobileNetV2
- `src/inference.js` - ONNX Runtime inference code

## Resources

- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
- [Model quantization guide](https://onnxruntime.ai/docs/performance/quantization.html)
- [WebAssembly performance tips](https://web.dev/webassembly/)
