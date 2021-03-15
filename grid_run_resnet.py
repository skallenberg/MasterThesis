import run
from utils.config import Config
import logging
import sys

logging.basicConfig(level=logging.INFO)

model_list = [
   # "ResNet18",
   # "ResNet34",
   "ResNet18_SD",
   "ResNet34_SD",
]

for model in model_list:
   logging.info("\n\n############## NEW MODEL ###################\n\n")
   Config().change_value("Setup", "Architecture", model)
   logging.info("Using Model:\t%s" % (model))
   try:
      run.train_model()
   except Exception as e:
      print(e)
      logging.info("Run Failed")

