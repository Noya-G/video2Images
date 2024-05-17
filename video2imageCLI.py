import signal
import sys
import os
from video2Images import signal_handler, run_odm
import frame_operation

signal.signal(signal.SIGINT, signal_handler.ctrlc_signal_handler)
# signal.signal(signal.SIGTSTP, signal_handler.ctrlz_signal_handler)

help_message = \
    "\nVideo2ImageCLI - Image Extraction Tool\n\n\
    ===============================================================\n\n\
    Usage:\n\
      [command] [options]\n\
    Commands:\n\
      extract           Extract frames from a video file.\n\
      help              Show this help message.\n\n\
    Options:\n\
      -v, --video       Path to the video file (required).\n\
      -o, --output      Output directory for extracted images.Default is images dir\n\
      -f, --format      Image format (e.g., png, jpg). Default is 'png'.\n\
      -r, --rate        Frame extraction rate (e.g., 1 frame every 5 seconds) Default is 5).\n\
      -m, --memory      Use memory to make sure the same photo is not taken twice Default is False\n\
      -c, --count       Extracting frames by quantity (will be averaged by length) \n\
      -l, --limit       Limiting the number of frames to be extracted \n\
      -p, --percent     Extracting frames by percentage difference (e.g,10- above this percentage difference)\n\
      -?, --help        Display this help message.\n\n\
    Examples:\n\
      # Extract one frame every 5 seconds from a video and save as PNG\n\
      extract --video=myvideo.mp4 --output=images --rate=5 --format=png\n\
      # Extract 100 frames from video and save it in images dir as png file\n\
      extract -v=myvideo.mp4 -o=images -c=100 -f=png\n\
      # Extract one frame every 1 seconds from a video asn save as png file in images dir.\n\
        In addition use in memory take frame only if there is a change above 10% with total limitation of 50. \n\
      extract -v=video.mp4 -o=images -f=png -r=1 -m=True -p=10 -l=50\n\
    Notes:\n\
      - Ensure the video file path is correct.\n\
      - Make sure the output directory is writable.\n\
      - The tool requires appropriate permissions to read and write files.\n\n\
    For more information, visit our documentation at [link to documentation].\n\n\
    ---\n\n\
    Feel free to adjust command names, options, and examples to align with your CLI tool's capabilities and design.\n\
    "
welcome_message = "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n" \
                  "Hi and welcome to Video2ImageCLI!\nThis is command line tool to processes image from video.\n" \
                  "Simply follow the prompts or use the --help option to learn more about the available commands.\n\n"
error_message = "There is no such command, try again or use help"
cli_message = "Video2ImageCLI>>> "
help_flags = ['help', '--help', '-help', '-h', '-?']
extract_flags = ['-v', '--video', '-o', ' --output', '-f', '--format', '-r', '--rate', '-m', '--memory',
                 '-c', '--count', '-l', '--limit', '-p', '--percent']

exit_flags = ['quit()', 'quit', 'exit']

int_val = ['-r', '-c', '-l', '-p']
bool_val = ['-m']


def parse_flag(flag_lst):
    if flag_lst[0] not in extract_flags:
        print(error_message)
        return 0, None
    reduce_flag = flag_lst[0].replace('-', '', 1) if len(flag_lst[0]) > 2 else flag_lst[0]
    key = reduce_flag[:2]
    value = flag_lst[1]
    if key in int_val:
        try:
            value = int(value)
        except ValueError as v_e:
            print(f"Invalid command. Check - {v_e}")
            value = None
    elif key in bool_val:
        if value in ['False', 'True']:
            value = bool(value)
        else:
            print(f"Expect to receive boolean value ['False', 'True'] but got - {value}")
            value = None
    return key, value


def parse_command(command_p):
    extract_option = {'-o': 'images', '-f': 'png', '-m': False}
    for option in command_p:
        opt_k_v = option.split('=')
        if len(opt_k_v) == 2:
            key, value = parse_flag(opt_k_v)
            if value is None:
                return
            extract_option[key] = value
        else:
            print(f"{error_message}. Check - {option}")
            return
    if extract_option.get('-v') is None:
        print("Need to insert path to input file, try again.")
        return
    if len(extract_option) > 0:
        return extract_option
    return


def command_management(command):
    _p = command.get('-p')
    _l = command.get('-l')
    _m = command.get('-m')

    _v = command['-v']
    if not os.path.isfile(_v):
        print(f"The file not exist. Enter correct path and try again")
        return
    _o = command['-o']
    if not os.path.isdir(_o):
        os.makedirs(_o)
    _f = command['-f']
    if command.get('-c') is not None:
        _c = command['-c']
        frame_operation.extract_frames(_v, _o, _c, _f)
    elif command.get('-r') is not None:
        _r = command['-r']
        frame_operation.extract_frames_by_time(_v, _o, _r, _f, _p, _l, _m)

    # after the image extract finished run ODM:
    run_data = f"./run.sh --project-path /datasets {_o}"
    run_odm.run_command(run_data)


def cli_engine():
    print(welcome_message)
    try:
        while True:
            command_input = input(cli_message)
            if command_input in help_flags:
                print(help_message)
                continue
            if command_input in exit_flags:
                print("Bye Bye...")
                sys.exit(0)
            command_lst = command_input.split(' ')
            if command_lst[0] == 'extract':
                command_dict = parse_command(command_lst[1:])
                if command_dict is not None:
                    command_management(command_dict)
            else:
                print(error_message)
    except EOFError as e:
        print("\nEnd of input received. Exiting.")


if __name__ == '__main__':
    cli_engine()
