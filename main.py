from utils.config import get_configs
from utils.noiser import add_noise, add_chatter

configs = get_configs('config.json')

add_noise(configs['audio_path'], configs['noised_save_path'], configs['noise_STD'], configs['sample_rate'])
add_chatter(configs['audio_path'], configs['chatter_save_path'], configs['chatter_path'], configs['sample_rate'], configs['chatter_volumn'])
