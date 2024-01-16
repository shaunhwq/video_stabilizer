# Video Stabilizer

A video stabilizer using Sift features in a night. Sadly I was unable to find a free stabilization app to help me stabilize my climbing videos the way I want it so I guess I gotta write one myself.

Basically this app finds the selected region in subsequent frames, and warps it back to the first frame using perspective transformation. Then we write the warped frames to the output file before and copy the audio over from the source.

To speed things up we apply a pyrDown to reduce resolution by 2, since most phone videos capture video at a high resolution anyway. Helps to reduce noise since it applies a Gaussian blur as well, probably a better matching outcome.

Before using...
1. Your cropped needs to have some interesting, unchanging features (the more the better the alignment)
2. You want to do video stabilization fixed to the selected crop
3. Object of interest is very far away or camera is panning or tilting without moving (homography assumption)
4. Cropped region must be in camera view throughout the entire video (unless you don't care about black regions)
5. Input video size should not be too small since we further downsize it, suspect that your final video might have some shaking effect due to poor matching.
6. Output might not a perfectly stable video, personally observed some temporal artifacts. Depends on window and input video.

## Installation

### Virtual Environment
```
conda create -n video_stabilizer_venv python=3.8
conda activate video_stabilier_venv
pip3 install -r requirements.txt
```

Also, make sure that ffmpeg is installed on your machine.

For mac users,
```
brew install ffmpeg
```
For others, refer to ffmpeg documentation.

### Running the script
```
python3 stabilizer.py -h
usage: stabilizer.py [-h] -i INPUT_PATH -o OUTPUT_PATH

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input_path INPUT_PATH
                        Path to input video
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Desired output path
```

e.g.

```
python3 stabilizer.py -i input.mp4 -o output.mp4
```

### Copying audio from original and use it on output

```
ffmpeg -i input.mp4 -q:a 0 -map a input_video_audio.mp3
ffmpeg -i stabilizer_output.mp4 -i input_video_audio.mp3 -c:v copy -c:a aac -strict experimental output.mp4
```
