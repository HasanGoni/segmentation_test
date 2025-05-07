import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SizePreservingPatchLayerONNX(nn.Module):
    """
    ONNX-compatible patch conversion layer with exact numerical consistency
    """
    def __init__(self, patch_size=256, min_overlap=32):
        super().__init__()
        self.patch_size = patch_size
        self.min_overlap = min_overlap
        self.register_buffer('weight_mask', self._create_weight_mask())
        
    def _create_weight_mask(self):
        """Creates gaussian weight mask with controlled precision"""
        # Use float32 explicitly for consistent precision
        x = torch.linspace(-1, 1, self.patch_size, dtype=torch.float32)
        y = torch.linspace(-1, 1, self.patch_size, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        # Use a more ONNX-friendly gaussian formula
        gaussian = torch.exp(-(xx.pow(2) + yy.pow(2)) / 1.5)
        return gaussian.to(torch.float32)

    def _calculate_grid(self, H, W):
        """Calculate grid with explicit float32 calculations"""
        # Convert to float32 for consistent precision
        H, W = float(H), float(W)
        patch_size = float(self.patch_size)
        min_overlap = float(self.min_overlap)
        
        # Calculate patches needed with explicit float32
        n_patches_h = math.ceil((H - min_overlap) / (patch_size - min_overlap))
        n_patches_w = math.ceil((W - min_overlap) / (patch_size - min_overlap))
        
        # Calculate strides with controlled precision
        stride_h = torch.tensor((H - patch_size) / max(n_patches_h - 1, 1), dtype=torch.float32)
        stride_w = torch.tensor((W - patch_size) / max(n_patches_w - 1, 1), dtype=torch.float32)
        
        return {
            'n_patches_h': int(n_patches_h),
            'n_patches_w': int(n_patches_w),
            'stride_h': stride_h.item(),
            'stride_w': stride_w.item()
        }

    def forward(self, x):
        B, C, H, W = x.shape
        grid = self._calculate_grid(H, W)
        
        patches = []
        locations = []
        
        for i in range(grid['n_patches_h']):
            for j in range(grid['n_patches_w']):
                # Use explicit float32 calculations
                h_start = int(round(i * grid['stride_h']))
                w_start = int(round(j * grid['stride_w']))
                
                # Ensure exact boundary handling
                h_start = min(h_start, H - self.patch_size)
                w_start = min(w_start, W - self.patch_size)
                
                patch = x[:, :,
                         h_start:h_start + self.patch_size,
                         w_start:w_start + self.patch_size]
                
                patches.append(patch)
                locations.append((h_start, w_start))
        
        patches = torch.stack(patches, dim=1)
        # Apply weight mask with controlled precision
        patches = patches * self.weight_mask.to(x.dtype).view(1, 1, 1, self.patch_size, self.patch_size)
        
        return patches, (locations, (H, W))

class SizePreservingPatchMergerONNX(nn.Module):
    """
    ONNX-compatible patch merging layer with numerical consistency
    """
    def __init__(self, patch_size=256):
        super().__init__()
        self.patch_size = patch_size
        
    def forward(self, patches, info):
        locations, (H, W) = info
        B, N, C, H_patch, W_patch = patches.shape
        
        # Initialize with explicit zeros in float32
        output = torch.zeros((B, C, H, W), dtype=patches.dtype, device=patches.device)
        weights = torch.zeros((B, 1, H, W), dtype=patches.dtype, device=patches.device)
        
        for idx, (h_start, w_start) in enumerate(locations):
            patch = patches[:, idx]
            h_end = min(h_start + self.patch_size, H)
            w_end = min(w_start + self.patch_size, W)
            
            # Accumulate with controlled precision
            output[:, :, h_start:h_end, w_start:w_end] += patch[:, :, :(h_end-h_start), :(w_end-w_start)]
            weights[:, :, h_start:h_end, w_start:w_end] += 1.0
        
        # Normalize with epsilon for numerical stability
        eps = 1e-8
        output = output / (weights + eps)
        return output

class ExactSizePatchNetworkONNX(nn.Module):
    """
    ONNX-compatible network ensuring numerical consistency
    """
    def __init__(self, base_model, patch_size=256, min_overlap=32):
        super().__init__()
        self.patch_maker = SizePreservingPatchLayerONNX(patch_size, min_overlap)
        self.base_model = base_model
        self.patch_merger = SizePreservingPatchMergerONNX(patch_size)
        
    def forward(self, x):
        # Ensure input is float32 for consistency
        x = x.to(torch.float32)
        original_size = x.shape
        
        patches, info = self.patch_maker(x)
        
        # Process patches with controlled batch reshaping
        B, N = patches.shape[:2]
        patches = patches.reshape(B * N, *patches.shape[2:])
        processed_patches = self.base_model(patches)
        processed_patches = processed_patches.reshape(B, N, *processed_patches.shape[1:])
        
        # Merge patches with numerical stability
        output = self.patch_merger(processed_patches, info)
        
        assert output.shape == original_size, f"Size mismatch: Input {original_size}, Output {output.shape}"
        return output

def compare_pytorch_onnx(model, input_tensor, onnx_path="model.onnx"):
    """
    Utility function to compare PyTorch and ONNX outputs
    """
    # PyTorch forward pass
    model.eval()
    with torch.no_grad():
        pytorch_output = model(input_tensor)

    # Export to ONNX
    torch.onnx.export(model, input_tensor, onnx_path, 
                     opset_version=12,
                     input_names=['input'],
                     output_names=['output'],
                     dynamic_axes={'input': {0: 'batch_size'},
                                 'output': {0: 'batch_size'}})

    # ONNX inference
    import onnxruntime
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.cpu().numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]

    # Compare results
    pytorch_numpy = pytorch_output.cpu().numpy()
    max_diff = np.max(np.abs(pytorch_numpy - ort_output))
    mean_diff = np.mean(np.abs(pytorch_numpy - ort_output))
    
    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'pytorch_output': pytorch_numpy,
        'onnx_output': ort_output
    } 