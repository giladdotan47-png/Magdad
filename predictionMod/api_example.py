import latestModelAPI

# loading the videos and converting to tensors
vids, labels = latestModelAPI.csvToTensors(r"C:\Users\TLP\Desktop\GoPro_Labled_Data.csv")

# initiating the model before each video
latestModelAPI.initVid()

# choosing a video
idx = 1000
vid, label = vids[1000], labels[1000]

for frame in vid:
    # loading the frames one by one
    print(latestModelAPI.predictFrame(frame))