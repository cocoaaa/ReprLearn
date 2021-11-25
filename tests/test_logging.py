import logging
from reprlearn.utils.misc import now2str
import datetime as dt
# call to `basicConfig` should come before any logging.<level> methods
# Only the first call to `basicConfig` will take an effect, and
# any subsequent call will act like no-ops
logging.basicConfig(filename='example.log',
                    filemode='w',
                    format='%(asctime)s -- %(message)s',
                    datefmt="%m/%d/%Y %I:%M%S %p",
                    level=logging.DEBUG)
logging.debug(now2str())
logging.debug('logging debug')
logging.info('logging info')
logging.warning('logging warning...')
logging.error('looing error')
