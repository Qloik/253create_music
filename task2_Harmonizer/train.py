import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from model import ChoirDataset, collate_fn
from data_processing import Config
import os
import json
import pickle

# ===== BASELINE TRAINING FUNCTIONS =====

def compute_basic_loss(outputs, targets, lengths, criterion):
    """Standard cross entropy loss for baseline model"""
    total_loss = 0
    voice_weights = {'soprano': 1.0, 'alto': 1.0, 'tenor': 1.0, 'bass': 1.2}
    
    for voice in ['soprano', 'alto', 'tenor', 'bass']:
        pred = outputs[voice].view(-1, outputs[voice].size(-1))
        target = targets[voice].view(-1)
        
        # Create mask for valid positions
        mask = torch.zeros_like(target, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i * outputs[voice].size(1):(i * outputs[voice].size(1) + length)] = True
        
        pred_masked = pred[mask]
        target_masked = target[mask]
        
        if len(target_masked) > 0:
            voice_loss = criterion(pred_masked, target_masked)
            total_loss += voice_weights[voice] * voice_loss
    
    return total_loss / len(voice_weights)

class BaselineTrainer:
    """Simple trainer for baseline model"""
    def __init__(self, model, vocab_builder, device):
        self.model = model
        self.vocab_builder = vocab_builder
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        
        self.training_history = {
            'train_losses': [],
            'val_losses': []
        }
    
    def train_epoch(self, train_loader):
        """Train one epoch with basic setup"""
        self.model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            melody = batch['melody'].to(self.device)
            chords = batch['chords'].to(self.device)
            targets = {voice: seq.to(self.device) for voice, seq in batch['targets'].items()}
            lengths = batch['lengths'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(melody, chords)
            loss = compute_basic_loss(outputs, targets, lengths, self.criterion)
            
            loss.backward()
            self.optimizer.step()
            epoch_losses.append(loss.item())
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: Loss {loss.item():.4f}")
        
        return np.mean(epoch_losses)
    
    def validate(self, val_loader):
        """Simple validation"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                melody = batch['melody'].to(self.device)
                chords = batch['chords'].to(self.device)
                targets = {voice: seq.to(self.device) for voice, seq in batch['targets'].items()}
                lengths = batch['lengths'].to(self.device)
                
                outputs = self.model(melody, chords)
                loss = compute_basic_loss(outputs, targets, lengths, self.criterion)
                val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def train_model(self, train_loader, val_loader, save_path):
        """Basic training loop"""
        print("Starting baseline training...")
        best_val_loss = float('inf')
        
        for epoch in range(Config.EPOCHS):
            print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.training_history['train_losses'].append(train_loss)
            self.training_history['val_losses'].append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'training_history': self.training_history,
                    'model_config': {
                        'note_vocab_size': len(self.vocab_builder.note_to_idx),
                        'chord_vocab_size': len(self.vocab_builder.chord_to_idx),
                        'hidden_dim': 256,
                        'num_layers': 2
                    }
                }, save_path)
                print(f"  Best model saved! Val loss: {val_loss:.4f}")
        
        print(f"\nBaseline training completed! Best validation loss: {best_val_loss:.4f}")
        return self.training_history

# ===== IMPROVED TRAINING FUNCTIONS =====

def compute_music_constraint_loss(outputs, targets, lengths, vocab_builder):
    """Music theory constraint loss - encourage better harmonic progressions"""
    device = outputs['soprano'].device
    constraint_loss = 0
    
    # Voice spacing constraints
    for i in range(len(lengths)):
        length = lengths[i]
        
        for t in range(1, length):  # Start from second time step
            # Get prediction probabilities for each voice
            soprano_probs = torch.softmax(outputs['soprano'][i, t], dim=-1)
            alto_probs = torch.softmax(outputs['alto'][i, t], dim=-1)
            tenor_probs = torch.softmax(outputs['tenor'][i, t], dim=-1)
            bass_probs = torch.softmax(outputs['bass'][i, t], dim=-1)
            
            # Simple voice spacing constraint - encourage reasonable intervals
            # This is a simplified version, can be more complex in practice
            interval_penalty = compute_interval_penalty(
                soprano_probs, alto_probs, tenor_probs, bass_probs, vocab_builder
            )
            constraint_loss += interval_penalty
    
    return constraint_loss / (len(lengths) * max(1, torch.mean(lengths.float()).item()))

def compute_interval_penalty(soprano_probs, alto_probs, tenor_probs, bass_probs, vocab_builder):
    """Calculate interval constraint penalty (simplified version)"""
    # Simplified interval constraint implementation
    # In real projects, more complex constraints based on music theory can be added
    
    # Encourage different notes in different voices (avoid excessive repetition)
    # Calculate similarity between voices, penalize excessive similarity
    
    similarities = []
    probs_list = [soprano_probs, alto_probs, tenor_probs, bass_probs]
    
    for i in range(len(probs_list)):
        for j in range(i+1, len(probs_list)):
            # Calculate probability distribution similarity (simplified KL divergence)
            similarity = torch.sum(probs_list[i] * probs_list[j])
            similarities.append(similarity)
    
    # Penalize overly similar voices
    avg_similarity = torch.mean(torch.stack(similarities))
    penalty = torch.relu(avg_similarity - 0.3)  # Penalize if similarity exceeds 0.3
    
    return penalty

def compute_coherence_loss(outputs, lengths):
    """Coherence loss - encourage smooth voice leading"""
    coherence_loss = 0
    
    for voice in ['soprano', 'alto', 'tenor', 'bass']:
        voice_output = outputs[voice]
        
        for i in range(len(lengths)):
            length = lengths[i]
            if length <= 1:
                continue
            
            # Calculate smoothness of adjacent time step predictions
            for t in range(length - 1):
                current_probs = torch.softmax(voice_output[i, t], dim=-1)
                next_probs = torch.softmax(voice_output[i, t+1], dim=-1)
                
                # Calculate KL divergence between adjacent time step probability distributions
                kl_div = torch.sum(current_probs * torch.log(current_probs / (next_probs + 1e-8) + 1e-8))
                coherence_loss += kl_div
    
    return coherence_loss / (len(lengths) * 4)  # 4 voices

def compute_improved_loss(outputs, targets, lengths, criterion, vocab_builder):
    """Enhanced loss calculation with music constraints"""
    
    # Baseline loss
    base_loss = compute_basic_loss(outputs, targets, lengths, criterion)
    
    # Music constraint loss
    music_loss = compute_music_constraint_loss(outputs, targets, lengths, vocab_builder)
    
    # Coherence loss
    coherence_loss = compute_coherence_loss(outputs, lengths)
    
    # Combined loss
    total_loss = base_loss + 0.1 * music_loss + 0.05 * coherence_loss
    
    return total_loss, {
        'base_loss': base_loss.item(),
        'music_loss': music_loss.item(),
        'coherence_loss': coherence_loss.item(),
        'total_loss': total_loss.item()
    }

class ImprovedTrainer:
    """Enhanced trainer with advanced training techniques"""
    def __init__(self, model, vocab_builder, device):
        self.model = model
        self.vocab_builder = vocab_builder
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Use AdamW optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler - use cosine annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=Config.EPOCHS,
            eta_min=Config.LEARNING_RATE * 0.1
        )
        
        # Training history
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'loss_components': [],
            'learning_rates': []
        }
    
    def train_epoch(self, train_loader):
        """Enhanced single epoch training"""
        self.model.train()
        epoch_losses = {
            'total': [],
            'base': [],
            'music': [],
            'coherence': []
        }
        
        for batch_idx, batch in enumerate(train_loader):
            melody = batch['melody'].to(self.device)
            chords = batch['chords'].to(self.device)
            targets = {voice: seq.to(self.device) for voice, seq in batch['targets'].items()}
            lengths = batch['lengths'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(melody, chords)
            
            # Calculate enhanced loss
            total_loss, loss_components = compute_improved_loss(
                outputs, targets, lengths, self.criterion, self.vocab_builder
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=Config.GRAD_CLIP)
            
            self.optimizer.step()
            
            # Record losses
            epoch_losses['total'].append(loss_components['total_loss'])
            epoch_losses['base'].append(loss_components['base_loss'])
            epoch_losses['music'].append(loss_components['music_loss'])
            epoch_losses['coherence'].append(loss_components['coherence_loss'])
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: "
                      f"Loss {loss_components['total_loss']:.4f} "
                      f"(Base: {loss_components['base_loss']:.4f}, "
                      f"Music: {loss_components['music_loss']:.4f}, "
                      f"Coherence: {loss_components['coherence_loss']:.4f})")
        
        return {k: np.mean(v) for k, v in epoch_losses.items()}
    
    def validate(self, val_loader):
        """Validation function"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                melody = batch['melody'].to(self.device)
                chords = batch['chords'].to(self.device)
                targets = {voice: seq.to(self.device) for voice, seq in batch['targets'].items()}
                lengths = batch['lengths'].to(self.device)
                
                outputs = self.model(melody, chords)
                loss, _ = compute_improved_loss(
                    outputs, targets, lengths, self.criterion, self.vocab_builder
                )
                val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def train_model(self, train_loader, val_loader, save_path):
        """Complete enhanced training workflow"""
        print("Starting improved training...")
        best_val_loss = float('inf')
        patience = 15  # Early stopping patience
        patience_counter = 0
        
        for epoch in range(Config.EPOCHS):
            print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_loss = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.training_history['train_losses'].append(train_metrics['total'])
            self.training_history['val_losses'].append(val_loss)
            self.training_history['loss_components'].append(train_metrics)
            self.training_history['learning_rates'].append(current_lr)
            
            print(f"Train Loss: {train_metrics['total']:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
            print(f"  Components - Base: {train_metrics['base']:.4f}, "
                  f"Music: {train_metrics['music']:.4f}, "
                  f"Coherence: {train_metrics['coherence']:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'train_loss': train_metrics['total'],
                    'training_history': self.training_history,
                    'model_config': {
                        'note_vocab_size': len(self.vocab_builder.note_to_idx),
                        'chord_vocab_size': len(self.vocab_builder.chord_to_idx),
                        'hidden_dim': 256,
                        'num_layers': 2
                    }
                }, save_path)
                print(f"  New best model saved! Val loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping triggered after {patience} epochs without improvement")
                    break
        
        # Save final model (even if not the best)
        final_save_path = save_path.replace('.pth', '_final.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': len(self.training_history['train_losses']),
            'val_loss': self.training_history['val_losses'][-1] if self.training_history['val_losses'] else float('inf'),
            'train_loss': self.training_history['train_losses'][-1] if self.training_history['train_losses'] else float('inf'),
            'training_history': self.training_history,
            'model_config': {
                'note_vocab_size': len(self.vocab_builder.note_to_idx),
                'chord_vocab_size': len(self.vocab_builder.chord_to_idx),
                'hidden_dim': 256,
                'num_layers': 2
            }
        }, final_save_path)
        print(f"Final model saved to: {final_save_path}")
        
        # Save training history
        history_path = save_path.replace('.pth', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Training history saved to: {history_path}")
        
        print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
        print(f"Saved files:")
        print(f"  - Best model: {save_path}")
        print(f"  - Final model: {final_save_path}")
        print(f"  - Training history: {history_path}")
        
        return self.training_history

# ===== MAIN TRAINING FUNCTIONS =====

def train_baseline_model():
    """Train baseline model from scratch"""
    from data_processing import load_and_split_data, create_folders
    from model import DeepChoirModel, VocabBuilder
    
    print("=== Training Baseline Model ===")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    create_folders()
    
    # Define save paths
    BASELINE_MODEL_PATH = Config.BASELINE_MODEL_PATH
    BASELINE_VOCAB_PATH = Config.BASELINE_VOCAB_PATH
    
    # Load data
    print("Loading and splitting data...")
    train_data, val_data, test_data = load_and_split_data()
    print(f"Data loaded: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab_builder = VocabBuilder()
    vocab_builder.build_vocab(train_data)
    vocab_builder.save_vocab(BASELINE_VOCAB_PATH)
    print(f"Vocabulary saved to {BASELINE_VOCAB_PATH}")
    print(f"Vocabulary sizes: Notes={len(vocab_builder.note_to_idx)}, Chords={len(vocab_builder.chord_to_idx)}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = ChoirDataset(train_data, vocab_builder)
    val_dataset = ChoirDataset(val_data, vocab_builder)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    print("Initializing baseline model...")
    model = DeepChoirModel(
        note_vocab_size=len(vocab_builder.note_to_idx),
        chord_vocab_size=len(vocab_builder.chord_to_idx),
        hidden_dim=256,
        num_layers=2
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Train
    trainer = BaselineTrainer(model, vocab_builder, device)
    history = trainer.train_model(train_loader, val_loader, BASELINE_MODEL_PATH)
    
    print("\nBaseline model training completed!")
    return history

def train_improved_model():
    """Train improved model (can start from baseline)"""
    from data_processing import load_and_split_data, create_folders
    from model import DeepChoirModel, VocabBuilder
    
    print("=== Training Improved Model ===")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    create_folders()
    
    # Define save paths
    IMPROVED_MODEL_PATH = Config.IMPROVED_MODEL_PATH
    IMPROVED_VOCAB_PATH = Config.IMPROVED_VOCAB_PATH
    
    # Load data
    print("Loading and splitting data...")
    train_data, val_data, test_data = load_and_split_data()
    print(f"Data loaded: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Load or build vocabulary
    baseline_vocab_path = "models/baseline_vocab.pkl"
    if os.path.exists(baseline_vocab_path):
        print(f"Loading existing vocabulary from {baseline_vocab_path}")
        with open(baseline_vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
        vocab_builder = VocabBuilder()
        vocab_builder.note_to_idx = vocab_data['note_to_idx']
        vocab_builder.chord_to_idx = vocab_data['chord_to_idx']
        vocab_builder.idx_to_note = vocab_data['idx_to_note']
        vocab_builder.idx_to_chord = vocab_data['idx_to_chord']
    else:
        print("Building new vocabulary...")
        vocab_builder = VocabBuilder()
        vocab_builder.build_vocab(train_data)
    
    # Save improved vocab
    vocab_builder.save_vocab(IMPROVED_VOCAB_PATH)
    print(f"Vocabulary saved to {IMPROVED_VOCAB_PATH}")
    print(f"Vocabulary sizes: Notes={len(vocab_builder.note_to_idx)}, Chords={len(vocab_builder.chord_to_idx)}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = ChoirDataset(train_data, vocab_builder)
    val_dataset = ChoirDataset(val_data, vocab_builder)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    print("Initializing improved model...")
    model = DeepChoirModel(
        note_vocab_size=len(vocab_builder.note_to_idx),
        chord_vocab_size=len(vocab_builder.chord_to_idx),
        hidden_dim=256,
        num_layers=2
    ).to(device)
    
    # Try to load baseline model for fine-tuning
    baseline_model_path = Config.BASELINE_MODEL_PATH
    if os.path.exists(baseline_model_path):
        try:
            checkpoint = torch.load(baseline_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded baseline model from {baseline_model_path} for fine-tuning")
        except Exception as e:
            print(f"Could not load baseline model: {e}")
            print("Starting training from scratch...")
    else:
        print("No baseline model found, starting from scratch...")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Train with improved methods
    trainer = ImprovedTrainer(model, vocab_builder, device)
    history = trainer.train_model(train_loader, val_loader, IMPROVED_MODEL_PATH)
    
    print("\nImproved model training completed!")
    print(f"To use the improved model for generation, update these paths:")
    print(f"  VOCAB_PATH = '{IMPROVED_VOCAB_PATH}'")
    print(f"  MODEL_PATH = '{IMPROVED_MODEL_PATH}'")
    
    return history
