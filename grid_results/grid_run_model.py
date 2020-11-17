import run
from utils.config import Config
import logging
import sys

logging.basicConfig(level=logging.INFO)

model_list = [
    "BaseNet34",
    "ResNet34",
    "DenseNet34",
    "FractalNet4",
    "MRN_Net34",
    "NiN_Net34",
    "FASMGNetTest1",
    "FASMGNetTest2",
    "MGNet34",
    "MG16",
    "PMG16",
]

for model in model_list:
    logging.info("\n\n############## NEW MODEL ###################\n\n")
    Config().change_value("Setup", "Architecture", model)
    logging.info("Using Model:\t%s" % (model))
    try:
        run.train_model()
    except:
        logging.info("Run Failed")

