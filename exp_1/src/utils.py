import omegaconf
from omegaconf import OmegaConf

def load_config(model) -> omegaconf.DictConfig:
    if model == 'bert-base-uncased':
        configs = OmegaConf.load('exp_1/configs/config1.yaml')
    elif model == 'ModernBERT-base':
        configs = OmegaConf.load('exp_1/configs/config2.yaml')
    return configs