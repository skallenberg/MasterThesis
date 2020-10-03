import run
from utils.config import Config
import logging
import sys

logging.basicConfig(level=logging.INFO)

"""model_list = [
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
]"""
model_list = ["BaseTest"]
MixedPrecision_Options = [True, False]
MixUp_Options = [True, False]
LabelSmoothing_Options = [True, False]
GhostBatchNorm_Options = [True, False]
CELU_Options = [True, False]
WhiteningBlock_Options = [True, False]
StrideToPooling_Options = [True, False]
Depthwise_Options = [True, False]
ConvolutionBias_Options = [True, False]
TwoConvsPerBlock_Options = [True, False]
PoolBeforeBN_Options = [True, False]

for model in model_list:
    logging.info("\n\n############## NEW MODEL ###################\n\n")
    Config().change_value("Setup", "Architecture", model)
    logging.info("Using Model:\t%s" % (model))

    for MixedPrec in MixedPrecision_Options:
        Config().change_value("Trainer", "MixedPrecision", MixedPrec)
        logging.info("Using MixedPrecision:\t%s" % (MixedPrec))

        for MUp in MixUp_Options:
            Config().change_value("Trainer", "MixUp", MUp)
            logging.info("Using MixUp:\t%s" % (MUp))

            for LS in LabelSmoothing_Options:
                Config().change_value("Trainer", "LabelSmoothing", LS)
                logging.info("Using LabelSmoothing:\t%s" % (LS))

                for GH in GhostBatchNorm_Options:
                    Config().change_value("Misc", "GhostBatchNorm", GH)
                    logging.info("Using GhostBatchNorm:\t%s" % (GH))

                    for C in CELU_Options:
                        Config().change_value("Misc", "UseCELU", C)
                        logging.info("Using CELU:\t%s" % (C))

                        for W in WhiteningBlock_Options:
                            Config().change_value("Misc", "WhiteningBlock", W)
                            logging.info("Using WhiteningPatches:\t%s" % (W))

                            for STP in StrideToPooling_Options:
                                Config().change_value("Misc", "StrideToPooling", STP)
                                logging.info("Using StrideToPooling:\t%s" % (STP))

                                for D in Depthwise_Options:
                                    Config().change_value("Misc", "Depthwise", D)
                                    logging.info("Using DepthwiseConvolutions:\t%s" % (D))

                                    for CB in ConvolutionBias_Options:
                                        Config().change_value("Misc", "ConvolutionBias", CB)
                                        logging.info("Using ConvolutionBias:\t%s" % (CB))

                                        for TCB in TwoConvsPerBlock_Options:
                                            Config().change_value("Misc", "TwoConvsPerBlock", TCB)
                                            logging.info("Using TwoConvsPerBlock:\t%s" % (TCB))

                                            for PBN in PoolBeforeBN_Options:
                                                Config().change_value("Misc", "PoolBeforeBN", PBN)
                                                logging.info(
                                                    "Using PoolBeforeBatchNorm:\t%s" % (PBN)
                                                )
                                                try:
                                                    run.train_model()
                                                except:
                                                    logging.info("Run Failed")

