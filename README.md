![video2images-high-resolution-logo-black-transparent](https://github.com/Noya-G/video2Images/assets/73538626/24219972-6fd6-4d4c-8907-6ef8ac4b2b8c)


# Overview

Welcome to Video2ImageCLI! This command-line tool extracts frames from video files and processes them using ODM (OpenDroneMap). After extracting and processing the frames, the tool merges the resulting meshes using CloudCompare. This README provides instructions on how to use the tool, its commands, options, and examples.

# Features

Extract frames from video files at specified intervals or based on count.
Process extracted frames using ODM.
Merge meshes using CloudCompare.
Customizable options for frame extraction and mesh merging.
User-friendly CLI interface with help and error messages.
Installation

# Clone the repository:

bash
Copy code
git clone <repository_url>
cd <repository_directory>
Install dependencies:

Ensure you have Python installed.
Install the required Python packages:
bash
Copy code
pip install -r requirements.txt
Install ODM and CloudCompare as per their respective installation instructions.
Set up executable permissions:

bash
Copy code
chmod +x run.sh
Usage

Run the CLI tool by executing the Python script:

bash
Copy code
python video2image_cli.py
Commands and Options
Commands:
extract: Extract frames from a video file.
help: Show the help message.
Options:

```
-v, --video: Path to the video file (required).
-o, --output: Output directory for extracted images. Default is images.
-f, --format: Image format (e.g., png, jpg). Default is png.
-r, --rate: Frame extraction rate (e.g., 1 frame every 5 seconds). Default is 5.
-m, --memory: Use memory to ensure the same photo is not taken twice. Default is False.
-c, --count: Extract frames by quantity (averaged by length).
-l, --limit: Limit the number of frames to be extracted.
-p, --percent: Extract frames by percentage difference (e.g., 10 - above this percentage difference).
-?, --help: Display the help message.
```

Examples
Extract one frame every 5 seconds from a video and save as PNG:

bash
Copy code
```
extract --video=myvideo.mp4 --output=images --rate=5 --format=png
```
Extract 100 frames from video and save them in the images directory as PNG files:

bash
Copy code
```
extract -v=myvideo.mp4 -o=images -c=100 -f=png
```
Extract one frame every second from a video, save as PNG in the images directory, use memory, and take frames only if there is a change above 10% with a total limitation of 50:

bash
Copy code
```
extract -v=video.mp4 -o=images -f=png -r=1 -m=True -p=10 -l=50
```
Notes
Ensure the video file path is correct.
Make sure the output directory is writable.
The tool requires appropriate permissions to read and write files.
Help Message
To display the help message, use the following command:

bash
Copy code
```
extract --help
Exiting the CLI
To exit the CLI, you can type:
```

bash
Copy code
```
quit
Logging
```
The tool logs its operations to video2image.log and also outputs messages to the console.

##Detailed Functionality

##Frame Extraction
The frame_operation.extract_frames function extracts frames based on the specified count.
The frame_operation.extract_frames_by_time function extracts frames at specified intervals and can use additional options like memory and percentage difference.
Mesh Processing
After frame extraction, the run_odm.run_command function is used to process the frames with ODM.
Once ODM processing is complete, the tool merges the resulting meshes using CloudCompare via a subprocess call.


## Authors
- [Noya Gendelman](https://github.com/Noya-G)
- [Bat-Ya Ashkenazy](https://github.com/batya1999)
- [Shlomo Pearl](https://github.com/shlomoPearl)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## Description
Video2ImageCLI is a command-line tool for extracting frames from a video file. It provides various options for customizing the extraction process such as frame rate, output format, and more.


