#video2Images

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

##Authors
- [Noya Gendelman](https://github.com/Noya-G)
- [Bat-Ya Ashkenazy](https://github.com/batya1999)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This updated README provides clear instructions on how to run the code and offers help in case users encounter any issues.
