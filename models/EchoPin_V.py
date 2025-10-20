import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# Reuse the same base model definition as AVENet
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'comparison', 'IS3'))
from models_lvs import base_models


class EchoPin_M(nn.Module):
    """CochAV: Dual-channel cochleagram input model based on AVENet structure.

    - Image branch: Same as AVENet (ResNet18)
    - Audio branch: ResNet18, first conv layer receives 2-channel input (left/right cochleagram), captures spatial audio cues
    - Forward output: (A, logits, Pos, Neg) aligned with AVENet
    - Supports initialization from IS3 pretrained weights
    """

    def __init__(self, args, pretrained_path=None):
        super(EchoPin_M, self).__init__()

        # Image encoder: Keep consistent with AVENet
        self.imgnet = base_models.resnet18(modal='vision', pretrained=True)

        # Audio encoder: Based on AVENet audio resnet18
        self.audnet = base_models.resnet18(modal='audio', pretrained=True)

        self.m = nn.Sigmoid()
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))

        self.epsilon = args.epsilon
        self.epsilon2 = args.epsilon2
        self.tau = 0.03
        self.trimap = args.tri_map
        self.Neg = args.Neg

        # Improved detection head: multi-scale feature extraction + better gradient flow
        self.det_head = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # Preserve more spatial information
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 4),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)
        
        # Detection head weight initialization: reset after module initialization to avoid being overwritten
        with torch.no_grad():
            self.det_head[-1].weight.data.normal_(0, 0.01)  # Small random weights
            init_bias = torch.tensor([0.5, 0.5, 0.3, 0.3])
            init_bias = torch.clamp(init_bias, 1e-4, 1 - 1e-4)
            self.det_head[-1].bias.data = torch.logit(init_bias)  # [cx, cy, w, h]
        
        # Load pretrained weights
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained_weights(pretrained_path)

    def forward(self, image: torch.Tensor, audio_coch_stereo: torch.Tensor, args, mode: str = 'val'):
        """Forward pass.

        Args:
            image: [B, 3, H, W]
            audio_coch_stereo: [B, 2, F, T] mel-spectrogram
        Returns:
            (A, logits, Pos, Neg) consistent with AVENet
        """
        B = image.shape[0]
        mask = (1 - 100 * torch.eye(B, B, device=image.device))

        # Image encoder
        img = self.imgnet(image)
        img = F.normalize(img, dim=1)

        # Audio encoder (stereo cochleagram)
        aud = self.audnet(audio_coch_stereo)
        aud = self.avgpool(aud).view(B, -1)
        aud = F.normalize(aud, dim=1)

        # Join: Completely consistent with AVENet
        A = torch.einsum('ncqa,nchw->nqa', [img, aud.unsqueeze(2).unsqueeze(3)]).unsqueeze(1)
        A0 = torch.einsum('ncqa,ckhw->nkqa', [img, aud.T.unsqueeze(2).unsqueeze(3)])

        # Trimap
        Pos = self.m((A - self.epsilon) / self.tau)
        if self.trimap:
            Pos2 = self.m((A - self.epsilon2) / self.tau)
            Neg = 1 - Pos2
        else:
            Neg = 1 - Pos

        Pos_all = self.m((A0 - self.epsilon) / self.tau)

        # Positive similarity
        sim1 = (Pos * A).view(*A.shape[:2], -1).sum(-1) / (Pos.view(*Pos.shape[:2], -1).sum(-1))
        # Across negatives
        sim = ((Pos_all * A0).view(*A0.shape[:2], -1).sum(-1) / Pos_all.view(*Pos_all.shape[:2], -1).sum(-1)) * mask
        sim2 = (Neg * A).view(*A.shape[:2], -1).sum(-1) / Neg.view(*Neg.shape[:2], -1).sum(-1)

        if self.Neg:
            logits = torch.cat((sim1, sim, sim2), 1) / 0.07
        else:
            logits = torch.cat((sim1, sim), 1) / 0.07

        # Predict normalized bbox
        pred_norm = torch.sigmoid(self.det_head(A))  # [B, 4] in (0,1)
        return A, logits, Pos, Neg, pred_norm
    
    def _load_pretrained_weights(self, pretrained_path: str):
        """Load model parameters from IS3 pretrained weights
        
        Args:
            pretrained_path: Pretrained weight file path (.tar or .pth file)
        """
        print(f"Loading pretrained weights: {pretrained_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Get pretrained weight dictionary
            if 'model_state_dict' in checkpoint:
                pretrained_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            else:
                pretrained_dict = checkpoint
            
            # Get current model state dictionary
            model_dict = self.state_dict()
            
            # Filter out mismatched layers
            filtered_dict = {}
            for k, v in pretrained_dict.items():
                # Remove 'module.' prefix (if exists)
                key = k.replace('module.', '') if k.startswith('module.') else k
                
                # Only load matching layers
                if key in model_dict and model_dict[key].shape == v.shape:
                    filtered_dict[key] = v
                # else:
                #     print(f"Skipping mismatched layer: {key} (shape: {v.shape if hasattr(v, 'shape') else 'N/A'})")
            
            model_dict.update(filtered_dict)
            
            self.load_state_dict(model_dict, strict=False)
                
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
            print("Will use randomly initialized weights")


class EchoPin_S(nn.Module):
    """EchoPin_S: Dual-channel mel-spectrogram input model based on AVENet structure.

    - Image branch: Same as AVENet (ResNet18)
    - Audio branch: ResNet18, first conv layer receives 2-channel input (left/right mel-spectrogram), captures spatial audio cues
    - Forward output: (A, logits, Pos, Neg) aligned with AVENet
    - Supports initialization from IS3 pretrained weights
    """

    def __init__(self, args, pretrained_path=None):
        super(EchoPin_S, self).__init__()

        # Image encoder: Keep consistent with AVENet
        self.imgnet = base_models.resnet18(modal='vision', pretrained=True)

        # Audio encoder: Based on AVENet audio resnet18, but first layer supports 2 channels
        self.audnet = base_models.resnet18(modal='audio', pretrained=True)
        # Expand first layer from 1 channel (or expected single channel) to 2 channels
        # Note: audio modality uses conv1_a instead of conv1
        if hasattr(self.audnet, 'conv1_a'):
            old_conv = self.audnet.conv1_a
            new_conv = nn.Conv2d(
                in_channels=2,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )
            # Weight initialization: copy/mean initialization, make 2nd channel consistent with 1st channel for stable transfer
            with torch.no_grad():
                if old_conv.weight.shape[1] == 1:
                    new_conv.weight.data[:, 0:1] = old_conv.weight.data.clone()
                    new_conv.weight.data[:, 1:2] = old_conv.weight.data.clone()
                else:
                    # If originally not 1 channel, do channel mean
                    mean_weight = old_conv.weight.data.mean(dim=1, keepdim=True)
                    new_conv.weight.data = mean_weight.repeat(1, 2, 1, 1)
                if old_conv.bias is not None:
                    new_conv.bias.data[:] = old_conv.bias.data
            self.audnet.conv1_a = new_conv

        self.m = nn.Sigmoid()
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))

        self.epsilon = args.epsilon
        self.epsilon2 = args.epsilon2
        self.tau = 0.03
        self.trimap = args.tri_map
        self.Neg = args.Neg

        # Improved detection head: multi-scale feature extraction + better gradient flow
        self.det_head = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # Preserve more spatial information
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 4),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)
        
        # Detection head weight initialization: reset after module initialization to avoid being overwritten
        with torch.no_grad():
            self.det_head[-1].weight.data.normal_(0, 0.01)  # Small random weights
            init_bias = torch.tensor([0.5, 0.5, 0.3, 0.3])
            init_bias = torch.clamp(init_bias, 1e-4, 1 - 1e-4)
            self.det_head[-1].bias.data = torch.logit(init_bias)  # [cx, cy, w, h]
        
        # Load pretrained weights
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained_weights(pretrained_path)

    def forward(self, image: torch.Tensor, audio_coch_stereo: torch.Tensor, args, mode: str = 'val'):
        """Forward pass.

        Args:
            image: [B, 3, H, W]
            audio_coch_stereo: [B, 2, F, T] left/right cochleagram dual channel
        Returns:
            (A, logits, Pos, Neg) consistent with AVENet
        """
        B = image.shape[0]
        mask = (1 - 100 * torch.eye(B, B, device=image.device))

        # Image encoder
        img = self.imgnet(image)
        img = F.normalize(img, dim=1)

        # Audio encoder (stereo cochleagram)
        aud = self.audnet(audio_coch_stereo)
        aud = self.avgpool(aud).view(B, -1)
        aud = F.normalize(aud, dim=1)

        # Join: Completely consistent with AVENet
        A = torch.einsum('ncqa,nchw->nqa', [img, aud.unsqueeze(2).unsqueeze(3)]).unsqueeze(1)
        A0 = torch.einsum('ncqa,ckhw->nkqa', [img, aud.T.unsqueeze(2).unsqueeze(3)])

        # Trimap
        Pos = self.m((A - self.epsilon) / self.tau)
        if self.trimap:
            Pos2 = self.m((A - self.epsilon2) / self.tau)
            Neg = 1 - Pos2
        else:
            Neg = 1 - Pos

        Pos_all = self.m((A0 - self.epsilon) / self.tau)

        # Positive similarity
        sim1 = (Pos * A).view(*A.shape[:2], -1).sum(-1) / (Pos.view(*Pos.shape[:2], -1).sum(-1))
        # Across negatives
        sim = ((Pos_all * A0).view(*A0.shape[:2], -1).sum(-1) / Pos_all.view(*Pos_all.shape[:2], -1).sum(-1)) * mask
        sim2 = (Neg * A).view(*A.shape[:2], -1).sum(-1) / Neg.view(*Neg.shape[:2], -1).sum(-1)

        if self.Neg:
            logits = torch.cat((sim1, sim, sim2), 1) / 0.07
        else:
            logits = torch.cat((sim1, sim), 1) / 0.07

        # Predict normalized bbox
        pred_norm = torch.sigmoid(self.det_head(A))  # [B, 4] in (0,1)
        return A, logits, Pos, Neg, pred_norm
    
    def _load_pretrained_weights(self, pretrained_path: str):
        """Load model parameters from IS3 pretrained weights
        
        Args:
            pretrained_path: Pretrained weight file path (.tar or .pth file)
        """
        print(f"Loading pretrained weights: {pretrained_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Get pretrained weight dictionary
            if 'model_state_dict' in checkpoint:
                pretrained_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            else:
                pretrained_dict = checkpoint
            
            # Get current model state dictionary
            model_dict = self.state_dict()
            
            # Filter out mismatched layers
            filtered_dict = {}
            for k, v in pretrained_dict.items():
                # Remove 'module.' prefix (if exists)
                key = k.replace('module.', '') if k.startswith('module.') else k
                
                # Only load matching layers
                if key in model_dict and model_dict[key].shape == v.shape:
                    filtered_dict[key] = v
                # else:
                #     print(f"Skipping mismatched layer: {key} (shape: {v.shape if hasattr(v, 'shape') else 'N/A'})")
            
            model_dict.update(filtered_dict)
            
            self.load_state_dict(model_dict, strict=False)
            
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
            print("Will use randomly initialized weights")
