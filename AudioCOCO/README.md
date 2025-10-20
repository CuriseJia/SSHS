### 🚀 Strat
```bash
# single process
python data_preprocess.py input.wav output.npy

# batch process
python data_preprocess.py input_dir output_dir --batch

# continue to convert
python data_preprocess.py input_dir output_dir --batch --skip-existing
```

### Using in Python
```python
from data_preprocess import CochleagramPreprocessor
from cochleargram_config import get_config

config = get_config('default')
preprocessor = CochleagramPreprocessor(**config)

result = preprocessor.process_single_file('input.wav', 'output.npy', 'npy')
```

## 🔧 Recommended Configuration

### Security configuration
```python
{
    'sr': 16000,              # Sampling rate
    'n_filters': 38,          # Number of filters
    'low_lim': 50,            # Low frequency limit (Hz)
    'hi_lim': 7500,           # High frequency limit (Hz) - below Nyquist frequency
    'sample_factor': 2,       # Sampling factor
    'downsample_factor': None,  # Downsampling factor - no downsampling used
    'nonlinearity': 'power',  # Power compression
    'strict': False,          # Mode
    'target_duration': 10.0   # Target duration (seconds) - used to standardize the temporal dimension of cochlear electrograms
}
```

## 📊 Features
-  🎵  Support. wav format audio files (mono and stereo)
-  🔧  Configurable parameters for generating cochlear electrograms
-  📊  Multiple output formats supported (. npy,. npz,. png)
-  📁  Batch processing support
-  📝  Detailed logging records
-  ⚡  Efficient batch processing optimization
-  🛠️  Automatic processing of stereo to mono conversion
-  📏  Intelligent Nyquist frequency adjustment
-  🛡️  Safe default configuration