import subprocess


def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    return_code = process.wait()
    if return_code != 0:
        print("Command failed with return code:", return_code)
