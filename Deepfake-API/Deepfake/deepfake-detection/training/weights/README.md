# Model Weights

This directory should contain the model weights file `model_weights.pth` for the ensemble deepfake detection models.

## For Development

For development and testing, you can create a symbolic link to your actual weights file:

### Windows
```cmd
mklink "model_weights.pth" "C:\path\to\your\actual\weights.pth"
```

### Linux/Mac
```bash
ln -s /path/to/your/actual/weights.pth model_weights.pth
```

## File Structure
```
training/
  weights/
    model_weights.pth  # Main weights file
    README.md          # This file
```

## Notes
- The weights file should be compatible with the models defined in the configuration.
- Ensure the file has the correct permissions to be read by the application.
