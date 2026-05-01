from ultralytics import YOLO
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

#This prevents a crash due to the window's OS attempting to run the code on each core of the CPU simultaneously rather than splitting tasks sequentially
if __name__ == '__main__':


    model = YOLO("yolo26n-obb.pt")

    #Train the model on the dataset annotated in roboflow
    results = model.train(
        data = "C:/Users/james/Desktop/Training Datasets/Robot Vision Project data V3 unuagmented but split.v2i.yolov8-obb/data.yaml",
        epochs = 300,
        imgsz = 1024,
        batch = 8,
        optimizer = "MuSGD",
        patience = 50,
        lr0 = 0.01,
        lrf = 0.01001,
        momentum = 0.93772,
        weight_decay = 0.00093,
        warmup_epochs = 2.40107,
        warmup_momentum = 0.85494,
        box = 7.90802,
        cls = 0.44164,
        dfl = 1.46099,
        hsv_h = 0.02011,
        hsv_s = 0.70641,
        hsv_v = 0.35329,
        degrees = 0.00242,
        translate = 0.11274,
        scale = 0.34583,
        shear = 0.0,
        perspective = 0.00033,
        flipud = 0.0,
        fliplr = 0.57586,
        bgr = 0.00741,
        mosaic = 1.0,
        mixup = 0.00167,
        cutmix = 8.0e-05,
        copy_paste = 0.00423,
        close_mosaic = 7,
        plots=True,
        save=True,
        val=True,
        name = "YOLO26_for_machine_vision",
        device = 0,
        workers = 4
    )
