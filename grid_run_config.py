import run
from utils.config import Config
import logging
import sys

logging.basicConfig(level=logging.INFO)

model_list = [
    "ResNet34",
    "ResNet50",
    "DenseNet34",
    "DenseNet121",
    "FractalNet4",
    "MRN_Net34",
    "FASMGNetTest1",
    "FASMGNetTest2",
    "MGNet34",
    "MG16",
    "PMG16",
]

MixUp_Options = [True, False]
LabelSmoothing_Options = [True, False]
Depthwise_Options = [True, False]
Optimizer_Options = ["SGD","Adam","AdaGrad", "AdamW"]
LR_Options = [0.1,0.01,0.001,0.0001]

for model in model_list:
    Config().change_value("Setup", "Architecture", model)
    
    for optimizer in Optimizer_Options:
        Config().change_value("Optimizer", "Type", optimizer)

        for lr in LR_Options:
            Config().change_value("Optimizer", "LearnRate", lr)

            for MUp in MixUp_Options:
                Config().change_value("Trainer", "MixUp", MUp)

                for LS in LabelSmoothing_Options:
                    Config().change_value("Trainer", "LabelSmoothing", LS)

                    for D in Depthwise_Options:
                        Config().change_value("Misc", "Depthwise", D)
                        logging.info("Using Model:\t%s" % (model))
                        logging.info("Using Optimizer:\t%s" % (optimizer))
                        logging.info("Using LearnRate:\t%s" % (lr))
                        logging.info("Using MixUp:\t%s" % (MUp))
                        logging.info("Using LabelSmoothing:\t%s" % (LS))
                        logging.info("Using DepthwiseConvolutions:\t%s" % (D))

                        try:
                            run.train_model()
                        except Exception as e:
                            print(e)
                            logging.info("Run Failed")

