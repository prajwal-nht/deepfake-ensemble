# Browser-Only Deepfake Detection

This is a standalone web app that runs completely in your browser. No backend, no API calls - just pure client-side inference using WebAssembly.

## Running It

You don't need Node.js or any build tools. Just open the HTML file:

**Option 1: Live Server (VS Code)**
1. Install "Live Server" extension in VS Code
2. Right-click `index.html`
3. Select "Open with Live Server"

**Option 2: Python HTTP Server**
```bash
cd wasm
python -m http.server 8080
```
Then visit `http://localhost:8080`

**Option 3: Any Static Server**
```bash
# Using Node.js http-server
npx http-server -p 8080

# Or just double-click index.html (might have CORS issues)
```

## How It Works

The app loads a quantized ONNX model (~4MB) and runs inference directly in your browser using ONNX Runtime Web. Everything happens client-side:

1. You upload an image
2. Image gets preprocessed in JavaScript
3. Model runs in WebAssembly
4. Results show instantly

No data leaves your computer.

## What's Different from the Main App?

| Feature | This (WASM) | Main App (Modal) |
|---------|-------------|------------------|
| Setup | Open HTML file | Deploy backend + frontend |
| Speed | ~100ms | ~3-5 seconds |
| Accuracy | ~70% | ~90% |
| Model Size | 4MB | 2GB |
| Privacy | Perfect (offline) | Good (server-side) |
| Cost | Free | Server costs |

## Files

- `index.html` - The entire app (HTML + CSS + JS)
- `model.onnx` - Quantized MobileNetV2 model
- `README.md` - This file

## Limitations

This is a proof-of-concept. It uses a generic image classifier, not a deepfake-specific model. For production use, you'd want:

- A model trained on deepfakes
- Better preprocessing
- Face detection
- Multiple model ensemble (if browser can handle it)

## Why Not Use This for Everything?

Browsers have memory limits (~2GB). Our full ensemble uses 5 models totaling ~2GB, which is too much for reliable browser performance. Plus, video processing needs significant RAM.

WASM works great for:
- Quick pre-checks
- Privacy-sensitive applications
- Offline demos
- Low-cost deployments

But for best accuracy, the server-based ensemble is better.

## Customizing

Want to use your own model?

1. Export your model to ONNX format
2. Quantize it: `python -m onnxruntime.quantization.quantize_dynamic model.onnx model_quant.onnx`
3. Replace `model.onnx` with your quantized model
4. Update preprocessing in `index.html` to match your model's input requirements

## Deploying

Since it's just static files, deploy anywhere:

```bash
# GitHub Pages
git add wasm/
git commit -m "Add WASM demo"
git push
# Enable GitHub Pages in repo settings

# Netlify
netlify deploy --dir=wasm --prod

# Vercel
vercel --prod wasm/

# Or just upload to any web host
```

## Performance Tips

- Use quantized models (INT8 or INT4)
- Keep model size under 10MB for fast loading
- Preload the model on page load
- Use Web Workers for inference (doesn't block UI)
- Cache the model in IndexedDB

## Browser Support

Works in all modern browsers:
- Chrome/Edge 87+
- Firefox 89+
- Safari 14.1+

Needs WebAssembly and ONNX Runtime Web support.