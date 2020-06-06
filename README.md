# alterate_affection  &mdash; a repurpose of [stylegan-encoder](https://github.com/Puzer/stylegan-encoder)
This repository tries to use [`Puzer/stylegan-encoder`](https://github.com/Puzer/stylegan-encoder) to change 
the affection of a video.

### Setup:

1) Using a docker
    * [shh into your remote server](https://gist.github.com/ralcant/7633cb0068a440f687bf4b75019fd5c5#shh-into-your-remote-server)
    * [Create a docker container and expose a port](https://gist.github.com/ralcant/7633cb0068a440f687bf4b75019fd5c5#create-a-docker-container): We will be using Jupyter Notebook, and therefore we need to expose a port from our docker container
    * [Once created, enter as root](https://gist.github.com/ralcant/7633cb0068a440f687bf4b75019fd5c5#enter-the-container)
    * If this is the first time using this docker, [install some generally useful packages](https://gist.github.com/ralcant/7633cb0068a440f687bf4b75019fd5c5#if-first-time-install-some-useful-packages) and [create your user](https://gist.github.com/ralcant/7633cb0068a440f687bf4b75019fd5c5#create-your-user)
    * Once you are in **root**, run the commands in [./preinstall.txt]
    * [Enter as yourself](https://gist.github.com/ralcant/7633cb0068a440f687bf4b75019fd5c5#enter-as-yourself). This will take you to `/home/ralcanta`. Replace `ralcanta` with your username.
    * `cd /u/ralcanta/`
2) Go to the folder of your choice and clone this repository and `cd` into it.
> git clone https://github.com/ralcant/alterate_affection.git
3) [Optional &mdash; but highly recommended] Working with Python virtual enviroments:
    * [Create a virtual enviroment](https://gist.github.com/ralcant/7633cb0068a440f687bf4b75019fd5c5#create-a-virtual-enviroment)
    * [Activate your virtual enviroment](https://gist.github.com/ralcant/7633cb0068a440f687bf4b75019fd5c5#activate-your-virtual-enviroment). 
    * [Make that virtual enviroment accessible in your Jupyter Notebook](https://gist.github.com/ralcant/7633cb0068a440f687bf4b75019fd5c5#make-that-virtual-enviroment-accessible-in-your-jupyter-notebook) 
3) [Start you Jupyter Notebook in remote server](https://gist.github.com/ralcant/7633cb0068a440f687bf4b75019fd5c5#start-your-jupyter-notebook-in-your-remote-server)
4) In your local computer, [create a ssh tunnel](https://gist.github.com/ralcant/7633cb0068a440f687bf4b75019fd5c5#creating-a-ssh-tunnel)
5) [Access the jupyter notebook locally](https://gist.github.com/ralcant/7633cb0068a440f687bf4b75019fd5c5#access-the-jupyter-notebook-locally) and start working on the [general_video_processing notebook](./general_video_processing.ipynb)! If you are using a virtual enviroment, **Make sure to choose it as the kernel of your notebook** before running anything.

### Working with a video:

Now let the *fun* begin!

After the setup, we can work with the [general_video_processing notebook](./general_video_processing.ipynb)
The notebook should be self explanatory, but in a general sense this is what happens:

0) **Getting everything ready** : Here we install the [necessary packages](./requirements.txt), and create any folder we might later need.
1) **Breaking the video into multiple frames** : Here we take a video and split it in multiple frames using cv2. We also store the fps (*frames per second*) of the video, which will be useful later.
2) **Updating every frame** : This is the *heavy* and most *time consuming* part of the whole notebook. The main goal is to update every frame with the person's emotion changed. This is where the `stylegan-encoder` code will be most useful. We divide the full work in subsections:
    * **2.1: Getting the aligned images out of every frame** : Here we use a modified version of stylegan-encoder's `align_images.py` code. Here we store, for every frame, the positions of the face in a variable called `ALL_ALIGNED_INFO`. This will be useful later.
    * **2.2: Generating the latent vectors from the aligned images** : This is by far the step that takes the most time (we are talking about *hour**s***). It uses stylegan-encoder's `encode_images.py`. The latent vectors will be useful to change the `affect` in every frame (see next step)
    * **2.3: Changing the affect of the *aligned* frames, and use this to change the affect of the *original* frames**: We use the latent vectors from `2.2` and stylegan-encoder's `smile_direction` to change the emotion of every aligned frame. Then we use the values from `ALL_ALIGNED_INFO` and some image processing to put that face into our original frame.
3) **Combining the processed frames into a video**: We use cv2 for this. The output will be video with no sound of the updated frames. For this to work we use the fps we found in Step #1
4) **Extracting the audio from the original video**: We use [moviepy](https://zulko.github.io/moviepy/) for this and we store the mp3 audio of our original video.
5) **Adding the audio to our processed video**: **Final step!** We use moviepy for this too.

Phew! That was quite a lot.

### Current limitations
* Still need to see a way to not harcode the face dimension
* Adding the **original** audio does not seem to be a good idea, as now the lips of the transformed frames are not in sync with the sound 

You can see the original readme [here](./PREVIOUS_README.md)