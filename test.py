from model import Segmentator
from config import update_config
from config import CONF

path_to_model = './meta_data/models/best.pth'
path_to_test = './data/val'

update_config(CONF, "./config.yaml")

model = Segmentator('cuda:2', CONF)

model.load_model(path_to_model)
hook = model.test(path_to_test, 4, save_dir="test_results")
