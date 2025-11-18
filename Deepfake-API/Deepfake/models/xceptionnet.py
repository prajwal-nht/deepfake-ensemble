import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from .xception import xception as create_xception

class FeatureExtractor(nn.Module):
    """
    Abstract class to be extended when supporting features extraction.
    It also provides standard normalized and parameters
    """

    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_trainable_parameters(self):
        return self.parameters()

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class Xception(FeatureExtractor):
    def __init__(self):
        super(Xception, self).__init__()
        self.xception = create_xception()
        self.xception.last_linear = nn.Linear(2048, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.xception.features(x)
        x = nn.ReLU(inplace=True)(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.xception.forward(x)

class SiameseTuning(FeatureExtractor):
    def __init__(self, feat_ext: FeatureExtractor, num_feat: int, lastonly: bool = True):
        super(SiameseTuning, self).__init__()
        self.feat_ext = feat_ext()
        if not hasattr(self.feat_ext, 'features'):
            raise NotImplementedError('The provided feature extractor needs to provide a features() method')
        self.lastonly = lastonly
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features=num_feat),
            nn.Linear(in_features=num_feat, out_features=1),
        )

    def features(self, x):
        x = self.feat_ext.features(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lastonly:
            with torch.no_grad():
                x = self.features(x)
        else:
            x = self.features(x)
        x = self.classifier(x)
        return x

    def get_trainable_parameters(self):
        if self.lastonly:
            return self.classifier.parameters()
        else:
            return self.parameters()

class XceptionST(SiameseTuning):
    def __init__(self):
        super(XceptionST, self).__init__(feat_ext=Xception, num_feat=2048, lastonly=True)

def load_model(device, model_filename):
    """
    Load the XceptionST model from a saved state dictionary.
    
    Args:
        device: torch.device to load the model onto
        model_filename: Path to the saved model file
    
    Returns:
        Loaded XceptionST model in evaluation mode
    """
    model = XceptionST().to(device)
    model.feat_ext.load_state_dict(torch.load(model_filename, map_location=device))
    model.eval()
    return model

def run_inference(model, input_tensor, device):
    """
    Run inference on the input tensor using the XceptionST model.
    
    Args:
        model: Loaded XceptionST model
        input_tensor: Input tensor of shape (batch_size, channels, height, width)
        device: torch.device to run inference on
    
    Returns:
        Probability as a float
    """
    with torch.no_grad():
        outputs = model(input_tensor.to(device))
        probs = torch.sigmoid(outputs).mean().item()
    return probs