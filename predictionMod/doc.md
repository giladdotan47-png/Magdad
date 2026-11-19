<h1>camera's setup</h1>
the tripod should be set 90cm from base of the architecture to the side.\ 
it should be noted that the bottle will hit in the middle of th architecture, which 120cm from the camera.\
while setting the camera, it is important that the camera is in the right angle such that it catches all bottle's trajectory

<h1>order of operation:</h1>
the user only runs "trainingControler.py"\
we first check whether we have new labeled data, if we do, then we add\
the new data to the already existing labeled data.\

after this, the user chooses the model he wishes to run

<br>
<br>

<h1>data saving convention:</h1>
<h2>training data</h2>

we are going to save the data in ".csv" file\
each type of data is saved into one file, the names are in "constants.py"\
each line of the csv correspond 

for each video we are going to preprocess it with
the image detection and collect 2 points

1) the geometrical center of battle
2) the geometrical center mass of the cap

in each frame we are going to store the data in the following way:\
each line is a different video
the first input is the video name, e.g. "001"
then, we are going to repeat the data from each frame in the following manner:
1. (int) x value of the cap
2. (int) y value of the cap
3. (int) z value of the cap
4. (int) x value of the bottle
5. (int) y value of the bottle
6. (int) z value of the bottle
7. (int) duration (1/framerate)

at the end of each line, we will end a 2 to mark the end of the video

if a video has an angle from only one camera, then we can only detect the movement\
through the xz plane, so we will assume that y=0 and put that in the input .csv file.

for the labels we are also going to save them into one ".csv" file\
we are going to save them in the following manner:
1. (str) video name, e.g. "001"
2. (int) z pred
3. (int) number of frames from the start of the video

***important notes***:
1. it is crutial that the same line in the csv will correspont to the same line in the labels file.\
for example: if we are saving the input data from vid number "002" in line 3,
then the labels from that video should also be in line 3 in the labels file
2. the input data from the frames should not pass the frame the label if from.
for example, if we aim for frame number 20, we should add the inputs from frames 1~19
3. the csv should not save the columns' names, that is, we only want to save the data.
<br>
<br>
<br>

<h2>labels - height and time</h2>
for this, we are going to create a ".csv" file with the following conventions:\
each row correspond to each video.\
each row will contain data in the following order:
1. the video's name
2. the height in the video (normilized to be in the range 0~1)
3. the frame number the target frame accrued
