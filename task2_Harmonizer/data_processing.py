import json
import os 
from music21 import corpus, chord, note, converter
from pathlib import Path
from sklearn.model_selection import train_test_split


class Config:
    # input data
    XML_FOLDER = "input/dataset1"     # xml files
    RAW_JSON = "input/rawdata.json"   # xml to json
    
    # model and vocab
    BASELINE_MODEL_PATH = "models/baseline_model.pth"
    BASELINE_VOCAB_PATH = "models/baseline_vocab.pkl"

    IMPROVED_MODEL_PATH = "models/improved_model.pth"
    IMPROVED_VOCAB_PATH = "models/improved_vocab.pkl"
    
    # ouput files
    MIDI_OUTPUT_DIR = "outputs/generated_chorales"   # mid file for listening
    EVAL_OUTPUT_DIR = "outputs/evaluation_results"   # save plot graphs

    
    # parameters for adjusting    
    BATCH_SIZE = 32   #64
    LEARNING_RATE = 0.0005  # 0.0001
    EPOCHS = 100
    MAX_LENGTH = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2

    WEIGHT_DECAY = 1e-5  
    GRAD_CLIP = 1.0  
    WARMUP_STEPS = 1000  
    
    # data split
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15



#  XML precess
def extract_notes_from_part(part):
    notes = []
    for elem in part.flat.notes:
        if isinstance(elem, note.Note):
            notes.append(elem.nameWithOctave)
        elif isinstance(elem, chord.Chord):
            root = elem.root()
            notes.append(root.nameWithOctave if root else elem.pitches[0].nameWithOctave)
    return notes

def extract_chords_measurewise(score):
    chords = []
    try:
        chordified = score.chordify()
        measures = chordified.getElementsByClass('Measure')
        for m in measures:
            chords_in_measure = m.recurse().getElementsByClass('Chord')
            if chords_in_measure:
                root = chords_in_measure[0].root()
                chords.append(root.name if root else chords_in_measure[0].pitches[0].name)
            else:
                notes_in_measure = m.recurse().getElementsByClass('Note')
                if notes_in_measure:
                    chords.append(notes_in_measure[0].name)
                else:
                    chords.append("N")
    except:
        chords = ["N"] * 10
    return chords

def extract_voices(score):
    parts = score.parts
    voices = []
    for i in range(4):
        if i < len(parts):
            voices.append(extract_notes_from_part(parts[i]))
        else:
            voices.append([])
    return voices[0], voices[1], voices[2], voices[3]

def process_xml_file(file_path):
    try:
        score = converter.parse(file_path)
        soprano, alto, tenor, bass = extract_voices(score)
        melody = soprano if soprano else extract_notes_from_part(score)
        chords = extract_chords_measurewise(score)
        
        all_sequences = [melody, chords, soprano, alto, tenor, bass]
        non_empty_sequences = [seq for seq in all_sequences if len(seq) > 0]
        
        if not non_empty_sequences:
            return None
        
        min_length = min(len(seq) for seq in non_empty_sequences)
        if min_length < 4:
            return None
        
        return {
            "filename": os.path.basename(file_path),
            "melody": melody[:min_length] if melody else ["N"] * min_length,
            "chords": chords[:min_length] if chords else ["N"] * min_length,
            "targets": {
                "soprano": soprano[:min_length] if soprano else ["N"] * min_length,
                "alto": alto[:min_length] if alto else ["N"] * min_length,
                "tenor": tenor[:min_length] if tenor else ["N"] * min_length,
                "bass": bass[:min_length] if bass else ["N"] * min_length
            }
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def convert_xml_to_json():
    print("Converting XML files to JSON...")
    folder_path = Path(Config.XML_FOLDER)
    if not folder_path.exists():
        print(f"Error: {Config.XML_FOLDER} not found")
        return False
    
    xml_files = []
    for ext in ['*.xml', '*.musicxml', '*.mxl']:
        xml_files.extend(folder_path.glob(ext))
    
    if not xml_files:
        print(f"No XML files found in {Config.XML_FOLDER}")
        return False
    
    data = []
    for xml_file in xml_files:
        sample = process_xml_file(xml_file)
        if sample:
            data.append(sample)
    
    if data:
        os.makedirs(os.path.dirname(Config.RAW_JSON), exist_ok=True)
        with open(Config.RAW_JSON, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Converted {len(data)} files to {Config.RAW_JSON}")
        return True
    return False



def create_folders():
    folders = [Config.MIDI_OUTPUT_DIR, Config.EVAL_OUTPUT_DIR,
               os.path.dirname(Config.IMPROVED_VOCAB_PATH),os.path.dirname(Config.BASELINE_VOCAB_PATH), os.path.dirname(Config.RAW_JSON)]
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

def load_and_split_data():
    with open(Config.RAW_JSON, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = data.get('train', data.get('data', list(data.values())[0]))
    
    valid_samples = []
    for sample in data:
        
        try:
            if all(key in sample for key in ['melody', 'chords', 'targets']):
                if all(voice in sample['targets'] for voice in ['soprano', 'alto', 'tenor', 'bass']):
                    valid_samples.append(sample)
        except:
            continue
    
    # split dataset
    train_data, temp_data = train_test_split(valid_samples, test_size=(Config.VAL_RATIO + Config.TEST_RATIO), random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=Config.TEST_RATIO/(Config.VAL_RATIO + Config.TEST_RATIO), random_state=42)
    
    print(f"Data split: Train {len(train_data)}, Val {len(val_data)}, Test {len(test_data)}")
    return train_data, val_data, test_data
