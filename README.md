## alterate_affection  &mdash; a repurpose of [stylegan-encoder](https://github.com/Puzer/stylegan-encoder)

# To change the affection of a video

### Setup:


### Working with a video
After the setup, we can work with the [general_video_processing notebook](./general_video_processing.ipynb)
The notebook should be self explanatory, but in a general sense this is what happens:

0) **Getting everything ready** : Here we install the [necessary packages](./requirements.txt), and create any folder we might later need.
1) **Breaking the video into multiple frames** : Here we take a video and split it in multiple frames using cv2. We also store the fps (*frames per second*) of the video, which will be useful later.
2) **Updating every frame** : This is the *heavy* and most *time consuming* part of the whole notebook. The main goal is to update every frame with the person's emotion changed. This is where the `stylegan-encoder` code will be most useful. We divide the full work in subsections:
    2.1) **Getting the aligned images out of every frame** : Here we use a modified version of stylegan-encoder's `align_images.py` code. Here we store, for every frame, the positions of the face in a variable called `ALL_ALIGNED_INFO`. This will be useful later.
    2.2) **Generating the latent vectors from the aligned images** : This is by far the step that takes the most time (we are talking about *hour**s***). It uses stylegan-encoder's `encode_images.py`. The latent vectors will be useful to change the `affect` in every frame (see next step)
    2.3) **Changing the affect of the *aligned* frames, and use this to change the affect of the *original* frames**: We use the latent vectors from `2.2` and stylegan-encoder's `smile_direction` to change the emotion of every aligned frame. Then we use the values from `ALL_ALIGNED_INFO` and some image processing to put that face into our original frame.
3) **Combining the processed frames into a video**: We use cv2 for this. The output will be video with no sound of the updated frames. For this to work we use the fps we found in Step #1
4) **Extracting the audio from the original video**: We use [moviepy](https://zulko.github.io/moviepy/) for this and we store the mp3 audio of our original video.
5) **Adding the audio to our processed video**: **Final step!** We use moviepy for this too.

Phew! That was quite a lot.

### Current limitations
* Still need to see a way to not harcode the face dimension
* Adding the **original** audio does not seem to be a good idea, as now the lips of the transformed frames are not in sync with the sound 

You can see the original readme [here](./PREVIOUS_README.md)