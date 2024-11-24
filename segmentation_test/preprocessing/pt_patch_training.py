import torch
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast
import gc

class MemoryOptimizedPatchModel(nn.Module):
    def __init__(self, 
                 base_model: nn.Module,
                 patch_layer: SizePreservingPatchLayerAdvancedBlending,
                 chunk_size: int = 4,  # Number of patches per chunk
                 use_checkpointing: bool = True):
        super().__init__()
        self.patch_layer = patch_layer
        self.base_model = base_model
        self.chunk_size = chunk_size
        self.use_checkpointing = use_checkpointing
        
        # Enable gradient checkpointing in base model if it's a supported architecture
        if use_checkpointing and hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()

    def _process_patch_chunk(self, patches: torch.Tensor) -> torch.Tensor:
        """Process a chunk of patches with optional checkpointing"""
        if self.use_checkpointing and self.training:
            return checkpoint(self.base_model, patches, preserve_rng_state=True)
        return self.base_model(patches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract patches
        with torch.no_grad():  # No need to track gradients during patch extraction
            patches, (locations, original_size) = self.patch_layer(x)
            torch.cuda.empty_cache()  # Clear memory after patch extraction
        
        B, N, C, H, W = patches.shape
        processed_patches = []
        
        # Process patches in chunks to manage memory
        for chunk_idx in range(0, N, self.chunk_size):
            chunk_end = min(chunk_idx + self.chunk_size, N)
            chunk = patches[:, chunk_idx:chunk_end].reshape(-1, C, H, W)
            
            # Process chunk
            with autocast(enabled=True):  # Use mixed precision for processing
                processed_chunk = self._process_patch_chunk(chunk)
            
            # Reshape back and store
            processed_chunk = processed_chunk.reshape(B, -1, *processed_chunk.shape[1:])
            processed_patches.append(processed_chunk)
            
            # Clear memory after each chunk
            del chunk
            torch.cuda.empty_cache()
            gc.collect()
        
        # Combine all processed patches
        processed_patches = torch.cat(processed_patches, dim=1)
        
        # Reconstruct with blending
        output = self.reconstruct_with_blending(processed_patches, locations, original_size)
        return output

    def reconstruct_with_blending(self, 
                                patches: torch.Tensor,
                                locations: List[Tuple[int, int]],
                                original_size: Tuple[int, int]) -> torch.Tensor:
        # Clear memory before reconstruction
        torch.cuda.empty_cache()
        
        with torch.cuda.amp.autocast(enabled=True):
            B, N, C, H, W = patches.shape
            output = torch.zeros((B, C, *original_size), device=patches.device)
            weight_acc = torch.zeros((B, 1, *original_size), device=patches.device)
            
            # Process reconstruction in chunks if the image is very large
            chunk_size = min(N, 16)  # Adjust based on available memory
            for idx in range(0, N, chunk_size):
                end_idx = min(idx + chunk_size, N)
                chunk_locations = locations[idx:end_idx]
                chunk_patches = patches[:, idx:end_idx]
                
                for j, (h_start, w_start) in enumerate(chunk_locations):
                    output[:, :, h_start:h_start + H, w_start:w_start + W] += \
                        chunk_patches[:, j] * self.patch_layer.weight_mask
                    weight_acc[:, :, h_start:h_start + H, w_start:w_start + W] += \
                        self.patch_layer.weight_mask
                
                # Clear memory after each reconstruction chunk
                del chunk_patches
                torch.cuda.empty_cache()
            
            # Final normalization
            output = output / (weight_acc + 1e-8)
            return output

class MemoryEfficientTrainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler()

    def train_step(self, batch: torch.Tensor) -> float:
        # Clear memory before each training step
        torch.cuda.empty_cache()
        gc.collect()
        
        self.optimizer.zero_grad()
        batch = batch.to(self.device)
        
        with torch.cuda.amp.autocast():
            output = self.model(batch)
            loss = self.criterion(output, batch)
        
        # Scale loss and backward pass
        self.scaler.scale(loss).backward()
        
        # Unscale gradients for clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step and update scaler
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Clear memory after backward pass
        del output
        torch.cuda.empty_cache()
        
        return loss.item()

# Example usage configuration
def configure_memory_efficient_training(
    base_model: nn.Module,
    patch_size: int = 256,
    min_overlap: int = 32,
    chunk_size: int = 4,
    blend_mode: str = 'gaussian'
) -> Tuple[MemoryOptimizedPatchModel, MemoryEfficientTrainer]:
    """Configure memory-efficient training setup"""
    
    # Initialize patch layer
    patch_layer = SizePreservingPatchLayerAdvancedBlending(
        patch_size=patch_size,
        min_overlap=min_overlap,
        blend_mode=blend_mode
    )
    
    # Create memory-optimized model
    model = MemoryOptimizedPatchModel(
        base_model=base_model,
        patch_layer=patch_layer,
        chunk_size=chunk_size,
        use_checkpointing=True
    )
    
    # Memory efficient training setup
    trainer = MemoryEfficientTrainer(
        model=model,
        criterion=nn.L1Loss(),
        optimizer=torch.optim.AdamW(model.parameters()),
        device=torch.device('cuda')
    )
    
    return model, trainer