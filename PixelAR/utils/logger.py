import logging

def create_logger(logging_dir=None, rank=0, debug=False):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0:  # real logger
        if logging_dir:
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        else:
            handlers=None
        logging.basicConfig(
            level=logging.INFO,
            format='[Rank 0][\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=handlers
        )
        logger = logging.getLogger(__name__)
    else:
        if logging_dir and debug:
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log_{rank}.txt")]
            logging.basicConfig(
                level=logging.INFO,
                format=f'[Rank {rank}][\033[34m%(asctime)s\033[0m] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers=handlers
            )
            logger = logging.getLogger(__name__)
            logger.addHandler(logging.NullHandler())
        else:
            logger = logging.getLogger(__name__)
            logger.addHandler(logging.NullHandler())
    return logger