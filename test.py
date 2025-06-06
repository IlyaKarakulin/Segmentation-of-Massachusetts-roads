from model import Segmentator

path_to_model = '/home/i.karakulin/Segmentation-of-Massachusetts-roads/meta_data/models/last.pth'
path_to_test = '/home/i.karakulin/Segmentation-of-Massachusetts-roads/data_tiff/val'

model = Segmentator('cuda:2')

model.load_model(path_to_model)
hook = model.test(path_to_test, 4, save_dir="/home/i.karakulin/Segmentation-of-Massachusetts-roads/test_results")
