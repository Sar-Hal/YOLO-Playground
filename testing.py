from ultralytics import YOLO # type: ignore
model = YOLO("yolo11x.pt")
results = model.track("videos\992624-hd_1920_1080_25fps.mp4",save=False,show=True)