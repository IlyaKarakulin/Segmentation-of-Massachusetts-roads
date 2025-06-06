from model import Segmentator

path_to_model = './meta_data/models/best.pth'
path_to_test = './data_tiff/test'

model = Segmentator('cuda:2')

model.load_model(path_to_model)
hook = model.test(path_to_test, 4, save_dir="test_results")
