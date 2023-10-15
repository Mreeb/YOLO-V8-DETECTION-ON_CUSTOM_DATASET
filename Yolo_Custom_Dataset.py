from ultralytics import YOLO
model = YOLO('best.pt')
results = model(source = 2, show=True, conf=0.6, save = True)