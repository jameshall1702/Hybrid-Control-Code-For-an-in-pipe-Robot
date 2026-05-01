from ultralytics import YOLO
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

#This prevents a crash due to the window's OS attempting to run the code on each core of the CPU simultaneously rather than splitting tasks sequentially
if __name__ == '__main__':

    #Choose size of model that will be used, the nano size is best for edge (mobile or small computer devices)
    model = YOLO("yolo26n-obb.pt")

    # Tune hyperparameters on my dataset for 30 epochs and 50 passes
    model.tune(
        data = "C:/Users/james/Desktop/Training Datasets/Robot Vision Project data V3 unuagmented but split.v2i.yolov8-obb/data.yaml",
        epochs = 30, #30 epochs for tuning is considered optimal for this size dataset as it allows sufficient time for patterns to develop and prevents wasting computation time
        iterations = 50, #As my dataset is medium to small 1200+ images before augments this is a lower number of iterations as the model is smaller and has a smaller number of images thus it provides
        batch = 16,
        optimizer = "MuSGD", #Despite MuSGD being advertised for YOLO26 for smaller dataset sizes (<50,000) and run times (our epochs and iterations are lower) AdamW can provide better results
        imgsz = 640, #640x640px image size is the optimal for yolo26 and it reduces computation time compared to larger image sizes without compromising accuracy from an image too small
        patience = 5, #This prevents the model generating poor sets of hyperparameters by stopping the iteration early if no positive change occurs for 5 epochs
        plots = False,#Plots and Save set to false prevent the algorithm generating and saving data for every iteration saving some GPU/CPU power but also saving storage on my laptop
        save = False,
        val = False,#This will be set to false to prevent mid training validation to speed up the process (if a large error between validation set results and test set results is noted it will be reinstated)
        device = 0, #Selects GPU as processing unit for the algorithm
        workers = 4, #Selects number of CPU cores that will feed data to the GPU this was found through trial and error as at this level of CPU usage the GPU runs at near 100% capacity with very limited throttling due to CPU data feed
    )
