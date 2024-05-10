import sys


def ctrlc_signal_handler(sig, frame):
    print("\nInterrupt received! Cleaning up...")
    sys.exit(0)


def ctrlz_signal_handler(sig, frame):
    print("\nSuspension detected! Preventing suspension...")