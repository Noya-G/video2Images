![video2images-high-resolution-logo-black-transparent](https://github.com/Noya-G/video2Images/assets/73538626/24219972-6fd6-4d4c-8907-6ef8ac4b2b8c)


This tool allows users to analyze a video file, extract frames, and select specific frames based on camera movement and other criteria.

## Features

- Frame extraction from a video file
- Analysis of camera movement between frames
- Selection of frames with significant camera movement
- Saving selected frames as image files

## Usage

1. **Installation**: Clone or download this repository to your local machine.

2. **Dependencies**: Make sure you have the following dependencies installed:
   - Python (version >= 3.6)
   - OpenCV (cv2)
   - NumPy
   - Concurrent futures

3. **Running the code**:

    - **Step 1**: Set up the environment:
        - Navigate to the directory where you have cloned or downloaded this repository.
        - Ensure you have Python and the required dependencies installed.

    - **Step 2**: Modify parameters (optional):
        - Open the `main.py` file.
        - Adjust parameters like `SKIP` and `THRESHOLD` according to your requirements.
        - Update the `video_path` variable to point to your input video file.

    - **Step 3**: Execute the code:
        - Run the `main.py` script using Python.
        - The selected frames will be saved in the output directory.

4. **Customization**: You can customize parameters like `SKIP` and `THRESHOLD` in the code to adjust frame selection criteria.

## Help

If you encounter any issues or need assistance, follow these steps:

- Make sure you have installed all the dependencies listed above.
- Check that the input video file path (`video_path`) is correctly set in the `main.py` file.
- Ensure that your Python environment is correctly configured.

## Additional Information

- The tool logs various information including the processing steps, git commit details, and frame selection criteria.
- Refer to the `LogMannager.py` module for logging configuration.
- Make sure to handle errors appropriately, especially if any dependencies are missing or if the input video file is not found.

## Authors
- [Noya Gendelman](https://github.com/Noya-G)
- [Bat-Ya Ashkenazy](https://github.com/batya1999)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## Description
Video2ImageCLI is a command-line tool for extracting frames from a video file. It provides various options for customizing the extraction process such as frame rate, output format, and more.


---

## Usage

### Commands
- **extract**: Extract frames from a video file.
- **help**: Show help message.

### Options
- `-v, --video`: Path to the video file (required).
- `-o, --output`: Output directory for extracted images. Default is 'images' directory.
- `-f, --format`: Image format (e.g., png, jpg). Default is 'png'.
- `-r, --rate`: Frame extraction rate (e.g., 1 frame every 5 seconds). Default is 5.
- `-m, --memory`: Use memory to ensure the same photo is not taken twice. Default is False.
- `-c, --count`: Extract frames by quantity (will be averaged by length).
- `-l, --limit`: Limit the number of frames to be extracted.
- `-p, --percent`: Extract frames by percentage difference (e.g., 10 - above this percentage difference).
- `-?, --help`: Display help message.

### Examples
- Extract one frame every 5 seconds from a video and save as PNG:
  ```
  extract --video=myvideo.mp4 --output=images --rate=5 --format=png
  ```
- Extract 100 frames from video and save them in the 'images' directory as PNG files:
  ```
  extract -v=myvideo.mp4 -o=images -c=100 -f=png
  ```
- Extract one frame every 1 second from a video, save as PNG, and use memory to avoid duplicates with a 10% change limitation and a total limitation of 50 frames:
  ```
  extract -v=video.mp4 -o=images -f=png -r=1 -m=True -p=10 -l=50
  ```

### Notes
- Ensure the video file path is correct.
- Make sure the output directory is writable.
- The tool requires appropriate permissions to read and write files.

For more information, visit our documentation at [link to documentation].

---

Feel free to adjust command names, options, and examples to align with your CLI tool's capabilities and design.
```
