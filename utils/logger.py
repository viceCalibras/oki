from logging.handlers import RotatingFileHandler
import logging

# Create a logger for the application-level messages
app_log = logging.getLogger('root.common')
app_log.setLevel(logging.DEBUG)
app_handler = RotatingFileHandler('logs/training.log', mode='a', maxBytes=5*1024*1024, backupCount=2, encoding=None, delay=0)
app_handler.setLevel(logging.DEBUG)
app_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s')
app_handler.setFormatter(app_formatter)
app_log.addHandler(app_handler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(app_formatter)
app_log.addHandler(consoleHandler)

if __name__ == "__main__":
    level = 30
    app_log.setLevel(level)
    app_log.critical("Level set to "+str(level))
    app_log.debug("Debug")
    app_log.info("Info")
    app_log.warn("Warn")