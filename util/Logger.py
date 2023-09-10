import os


class Logger:
    def __init__(self, basedir):
        self.logfile = os.path.join(basedir, "log.txt")

    def log(self, msg, out=False):
        with open(self.logfile, "a+") as logfile:
            logfile.write(msg)
            logfile.write("\n")
        if out:
            print(msg)

    def logo(self, msg):
        self.log(str(msg), True)
