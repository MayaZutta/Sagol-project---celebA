from dataclasses import dataclass

@dataclass
class trainConfig():
    root_dir: str = r"C:\Users\Owner\Desktop\SagolProject_Galit\CelebA" # should i chnage them to the uni comp? where?
    data_dir: str = r"C:\Users\Owner\Desktop\SagolProject_Galit\CelebA"
    exp_name: str  = 'exp_try_2'
    epochs: int = 100
