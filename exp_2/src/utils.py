import omegaconf
from omegaconf import OmegaConf

def load_config() -> omegaconf.DictConfig:
    configs = OmegaConf.load('exp_2/configs/config.yaml')
    return configs