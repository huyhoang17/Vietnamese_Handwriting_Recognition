import logging


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # formatter
    fmt = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler('spam.log', mode='a')
    file_handler.setLevel(logging.DEBUG)
    handler.setFormatter(fmt)
    file_handler.setFormatter(fmt)

    # add handler to formatter
    logger.addHandler(handler)
    logger.addHandler(file_handler)

    return logger
