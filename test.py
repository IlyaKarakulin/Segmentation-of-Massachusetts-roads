from model.model import RoadSegmentation

path_to_model = './best.pth'
path_to_test = './dataset/test'

model = RoadSegmentation('cuda:1')

model.load_model(path_to_model)
hook = model.test(path_to_test, 1, threshold=0.3, save_dir="./test_results")
