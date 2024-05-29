import logging
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    def __init__(self):
        super().__init__()

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception as e:
            self.handleError(record)
            self.handleError(e)


def create_tqdm_bar(iterable=None, **tqdm_kwargs):
    if iterable is not None:
        return tqdm(iterable, **tqdm_kwargs)
    else:
        return tqdm(**tqdm_kwargs)
