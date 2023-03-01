import logging

class Logger():
    def __init__(self, logfile='output.log'):
        self.logfile = logfile
        self.logger = logging.getLogger(logfile)
        fh = logging.FileHandler(logfile, mode='w')
        lf = logging.Formatter('[%(asctime)s] - %(message)s')
        fh.setFormatter(lf)
        self.logger.addHandler(fh)
        logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y_%m_%d %H:%M:%S',
            level=logging.INFO,
            # filename=self.logfile
        )

    def info(self, msg, *args):
        msg = str(msg)
        if args:
            # print(msg % args)
            self.logger.info(msg, *args)
        else:
            # print(msg)
            self.logger.info(msg)
