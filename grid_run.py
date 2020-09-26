import run


model_list = [
    "BaseNetTest",
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
]
epoch_list = [5, 50, 150, 200]
dataset_list = ["cifar10", "cifar100"]
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

