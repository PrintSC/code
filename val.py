import os
from models import YOLO

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':
    model_path = r""
    model = YOLO(model_path)
    train_model = model.val()
