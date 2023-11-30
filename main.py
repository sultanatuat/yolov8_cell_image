#from clearml import Task
#task = Task.init(project_name="my project", task_name="my task")

from ultralytics import YOLO
from torchsummary import summary
import glob
 
# Load the model.
model = YOLO('yolov8n.pt')
#model = YOLO()
epoch = 500
# Training.
results = model.train(
   data='custom_data.yaml',
   imgsz=640,
   epochs=epoch,
   batch=4,
   name='epoch_' + str(epoch))

# get image files
image_files = glob.glob("/home/rebeka/TUAT_workspace/source_codes/yolov8/datasets/valid/images/" + "*")

# Load the model.
model = YOLO('./runs/detect/epoch_' + str(epoch) + '/weights/best.pt')

for i in range(len(image_files)):
    model.predict(image_files[i], save=True)

## Evaluate the model's performance on the validation set
#results = model.val()

# # Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg')

#detection_results = model.predict("./datasets/valid/images/20230510_1805 20倍 タンチョウ肝臓AA.bmp", save=True)
# detection_results = detection_results[0]
# boxes = len(detection_results.boxes)
# box = detection_results.boxes[0]

# Export the model to ONNX format
#success = model.export(format='onnx', dynamic=True)