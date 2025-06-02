import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset
import pickle
# ===== model =====
class DeepChoirModel(nn.Module):
    def __init__(self, note_vocab_size, chord_vocab_size, hidden_dim=256, num_layers=2):
        super(DeepChoirModel, self).__init__()
        self.note_embed = nn.Embedding(note_vocab_size, hidden_dim)
        self.chord_embed = nn.Embedding(chord_vocab_size, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, batch_first=True)
        self.decoders = nn.ModuleDict({
            voice: nn.Sequential(
                nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True),
                nn.Linear(hidden_dim, note_vocab_size)
            ) for voice in ['soprano', 'alto', 'tenor', 'bass']
        })
    
    def forward(self, melody, chords):
        melody_embed = self.note_embed(melody)
        chords_embed = self.chord_embed(chords)
        combined = torch.cat([melody_embed, chords_embed], dim=-1)
        enc_out, _ = self.encoder(combined)
        
        outputs = {}
        for voice, decoder in self.decoders.items():
            dec_out, _ = decoder[0](enc_out)
            logits = decoder[1](dec_out)
            outputs[voice] = logits
        return outputs

class VocabBuilder:
    def __init__(self):
        self.note_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.chord_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_note = {0: '<PAD>', 1: '<UNK>'}
        self.idx_to_chord = {0: '<PAD>', 1: '<UNK>'}
    
    def build_vocab(self, data):
        note_counter = Counter()
        chord_counter = Counter()
        
        for sample in data:
            for note in sample['melody']:
                note_counter[note] += 1
            for voice_notes in sample['targets'].values():
                for note in voice_notes:
                    note_counter[note] += 1
            for chord in sample['chords']:
                chord_counter[chord] += 1
        
        for note, count in note_counter.items():
            if count > 1 and note not in self.note_to_idx:
                idx = len(self.note_to_idx)
                self.note_to_idx[note] = idx
                self.idx_to_note[idx] = note
        
        for chord, count in chord_counter.items():
            if count > 0 and chord not in self.chord_to_idx:
                idx = len(self.chord_to_idx)
                self.chord_to_idx[chord] = idx
                self.idx_to_chord[idx] = chord
    
    def encode_sequence(self, sequence, vocab_dict):
        return [vocab_dict.get(item, vocab_dict['<UNK>']) for item in sequence]
    
    def save_vocab(self, path):
        vocab_data = {
            'note_to_idx': self.note_to_idx,
            'chord_to_idx': self.chord_to_idx,
            'idx_to_note': self.idx_to_note,
            'idx_to_chord': self.idx_to_chord
        }
        with open(path, 'wb') as f:
            pickle.dump(vocab_data, f)



class ChoirDataset(Dataset):
    def __init__(self, data, vocab_builder, max_length=128):
        self.data = data
        self.vocab = vocab_builder
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        melody = self.vocab.encode_sequence(sample['melody'], self.vocab.note_to_idx)
        chords = self.vocab.encode_sequence(sample['chords'], self.vocab.chord_to_idx)
        
        voices = {}
        for voice_name, voice_data in sample['targets'].items():
            voices[voice_name] = self.vocab.encode_sequence(voice_data, self.vocab.note_to_idx)
        
        seq_len = min(len(melody), self.max_length)
        melody = melody[:seq_len] + [0] * (self.max_length - seq_len)
        chords = chords[:seq_len] + [0] * (self.max_length - seq_len)
        
        for voice_name in voices:
            voice_seq = voices[voice_name][:seq_len] + [0] * (self.max_length - seq_len)
            voices[voice_name] = voice_seq
        
        return {
            'melody': torch.tensor(melody, dtype=torch.long),
            'chords': torch.tensor(chords, dtype=torch.long),
            'targets': {voice: torch.tensor(seq, dtype=torch.long) 
                       for voice, seq in voices.items()},
            'length': seq_len
        }


def collate_fn(batch):
    melody = torch.stack([item['melody'] for item in batch])
    chords = torch.stack([item['chords'] for item in batch])
    targets = {}
    for voice in ['soprano', 'alto', 'tenor', 'bass']:
        targets[voice] = torch.stack([item['targets'][voice] for item in batch])
    lengths = torch.tensor([item['length'] for item in batch])
    return {'melody': melody, 'chords': chords, 'targets': targets, 'lengths': lengths}