from dataclasses import dataclass
import datetime

@dataclass
class trainConfig():
    root_dir: str = r"/galitylab/data/celeba" 
    data_dir: str = r"/galitylab/data/celeba"
    current_time = datetime.datetime.now()
    exp_name: str = f'exp_try_{current_time.strftime("%m%d_%H%M")}'
    epochs: int = 100
    
