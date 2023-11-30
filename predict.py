from ultralytics import YOLO
import glob

# get image files
image_files = glob.glob("/home/rebeka/TUAT_workspace/source_codes/yolov8/datasets/valid/images/" + "*")

# Load the model.
model = YOLO('./runs/detect/with_pretrain/epoch_10000/weights/best.pt')

for i in range(len(image_files)):
    model.predict(image_files[i], save=True)