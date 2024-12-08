from ultralytics import YOLO # type: ignore
model = YOLO("yolo11n.pt")
results = model.track("3904994-hd_1920_1080_30fps.mp4",save=True,show=True)