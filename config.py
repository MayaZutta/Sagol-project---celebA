from dataclasses import dataclass
import datetime

@dataclass
class trainConfig():
    root_dir: str = r"C:\Users\Owner\Desktop\SagolProject_Galit\CelebA" # should i chnage them to the uni comp? where?
    data_dir: str = r"C:\Users\Owner\Desktop\SagolProject_Galit\CelebA"
    current_time = datetime.datetime.now()
    exp_name: str = f'exp_try_{current_time.strftime("%m%d_%H%M")}'
    epochs: int = 100
    
