from dotenv import load_dotenv
load_dotenv()
import os
os.environ.setdefault("KAGGLE_USERNAME", os.environ.get("KAGGLE_USERNAME", ""))
os.environ.setdefault("KAGGLE_KEY", os.environ.get("KAGGLE_API_TOKEN", ""))

import kagglehub, shutil
path = kagglehub.dataset_download('ritwikakancharla/nemotron-math-v2-sft-high-medium-tools')
print('Downloaded to:', path)
os.makedirs('data/nemotron-sft-high-medium-tools', exist_ok=True)
shutil.copy(os.path.join(path, 'data.parquet'), 'data/nemotron-sft-high-medium-tools/')
print('Done')
