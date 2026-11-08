from ultralytics import YOLO

model = YOLO("yolo26m.pt")

model.train(
    data="ul://hound-31/datasets/shuttlecock-dataset",
    epochs=100,
    imgsz=640,
    batch=8,
    patience=20,
    project="badminton",
    name="shuttle_detector"
)

# results = model("./shot1.png")
# results[0].save(filename="shot1_annotated.png")