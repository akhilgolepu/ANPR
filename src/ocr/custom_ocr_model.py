"""
Custom OCR Model: CNN + LSTM with CTC Loss
Optimized for Indian License Plate Recognition
Architecture: ResNet18 backbone + LSTM + CTC decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


class CustomOCRModel(nn.Module):
    """
    Custom lightweight OCR model for license plate recognition.
    
    Architecture:
    - Input: 128x64 plate image (grayscale or RGB)
    - CNN: ResNet18 for feature extraction (2048 features)
    - RNN: 2-layer LSTM for sequence modeling
    - Output: Sequence of character logits (CTC compatible)
    
    Indian plates have 9-10 characters: AA-DD-AA-DDDD format
    """
    
    def __init__(self, num_classes=37, input_height=64, input_width=128):
        """
        Args:
            num_classes: 37 = 26 letters + 10 digits + blank char + space
            input_height: 64 pixels
            input_width: 128 pixels
        """
        super(CustomOCRModel, self).__init__()
        
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        
        # ============= CNN Backbone =============
        # Use ResNet18 pre-trained on ImageNet
        resnet18 = models.resnet18(pretrained=True)
        
        # Modify first conv layer for grayscale input (if needed)
        # Keep 3-channel for compatibility, but can convert images
        self.cnn = nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            resnet18.layer4,
            nn.AdaptiveAvgPool2d((1, 4))  # Output: (batch, 512, 1, 4)
        )
        
        # CNN output shape: (batch, 512, 1, 4)
        cnn_output_height = 1
        cnn_output_width = 4
        cnn_output_channels = 512
        
        # Reshape for LSTM: (batch, sequence_length, features)
        # sequence_length = width * height = 4 * 1 = 4
        # features = channels = 512
        self.cnn_output_channels = cnn_output_channels
        self.cnn_output_width = cnn_output_width
        
        # ============= RNN Layers =============
        lstm_input_size = cnn_output_channels * cnn_output_height
        lstm_hidden_size = 256
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )
        
        # Output linear layer: LSTM hidden -> character logits
        # LSTM output: (batch, seq_len, 512) because bidirectional
        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch, 3, 64, 128) - RGB image
        
        Returns:
            logits: (batch, seq_len, num_classes) - CTC compatible
        """
        # ===== CNN Forward =====
        # Input: (batch, 3, 64, 128)
        cnn_out = self.cnn(x)  # (batch, 512, 1, 4)
        
        # Reshape for LSTM
        batch_size = cnn_out.size(0)
        # (batch, 512, 1, 4) -> (batch, 4, 512)
        cnn_out = cnn_out.permute(0, 3, 1, 2).contiguous()
        cnn_out = cnn_out.view(batch_size, self.cnn_output_width, -1)
        
        # ===== LSTM Forward =====
        lstm_out, _ = self.lstm(cnn_out)  # (batch, seq_len, 512)
        
        # ===== Output Layer =====
        logits = self.fc(lstm_out)  # (batch, seq_len, num_classes)
        
        return logits


class CharacterMapping:
    """Map characters to indices and vice versa for Indian license plates"""
    
    def __init__(self):
        # Characters in order: A-Z, 0-9, space
        self.characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
        self.char_to_idx = {c: i for i, c in enumerate(self.characters)}
        self.idx_to_char = {i: c for i, c in enumerate(self.characters)}
        self.blank_idx = len(self.characters)
    
    def encode(self, text):
        """Convert text to indices"""
        text = text.upper().strip()
        indices = [self.char_to_idx.get(c, self.blank_idx) for c in text]
        return indices
    
    def decode(self, indices):
        """Convert indices back to text (simple, no CTC decode)"""
        text = ''.join(
            self.idx_to_char.get(idx, '') 
            for idx in indices 
            if idx != self.blank_idx
        )
        return text.strip()
    
    @property
    def num_classes(self):
        return len(self.characters) + 1  # +1 for blank (CTC)


class CTCLoss(nn.Module):
    """CTC Loss for sequence-to-sequence alignment"""
    
    def __init__(self, blank=36):
        super(CTCLoss, self).__init__()
        self.loss = nn.CTCLoss(blank=blank, reduction='mean')
    
    def forward(self, logits, targets, input_lengths, target_lengths):
        """
        Args:
            logits: (batch, seq_len, num_classes) from model
            targets: (sum(target_lengths),) flattened target indices
            input_lengths: (batch,) lengths of input sequences
            target_lengths: (batch,) lengths of target sequences
        
        Returns:
            loss: scalar
        """
        # CTC expects (seq_len, batch, num_classes) and log probabilities
        logits = logits.permute(1, 0, 2)  # (batch, seq_len, num_classes) -> (seq_len, batch, num_classes)
        logits = F.log_softmax(logits, dim=2)  # Apply log softmax
        loss = self.loss(logits, targets, input_lengths, target_lengths)
        return loss


def create_model(num_classes=37, pretrained=True):
    """Factory function to create model"""
    model = CustomOCRModel(num_classes=num_classes)
    return model


if __name__ == '__main__':
    # Test model
    model = CustomOCRModel(num_classes=37)
    x = torch.randn(2, 3, 64, 128)  # 2 samples, RGB, 64x128
    
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")  # Should be (2, seq_len, 37)
    
    # Test character mapping
    char_map = CharacterMapping()
    text = "TS09AB1234"
    encoded = char_map.encode(text)
    decoded = char_map.decode(encoded)
    print(f"Original: {text}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")
    print(f"Num classes: {char_map.num_classes}")
