import run
from utils.config import Config
import logging
import sys

logging.basicConfig(level=logging.INFO)

model_list = [
    "BaseTest",
    "BaseNet34",
    "ResBaseTest",
    "ResNet34",
    "DenseTest",
    "DenseNet34",
    "FractalTest",
    "FractalNet4",
    "MRN_BaseTest",
    "MRN_Net34",
    "NiN_BaseTest",
    "NiN_Net34",
    "MGNetTest",
    "FASMGNetTest1",
    "FASMGNetTest2",
    "MGNet34",
    "MAN_Test",
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

