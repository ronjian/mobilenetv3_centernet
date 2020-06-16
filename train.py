from lib.helper.logger import logger
from lib.core.base_trainer.net_work import trainner
import setproctitle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logger.info('train start')
setproctitle.setproctitle("detect")

trainner=trainner()

trainner.train()
