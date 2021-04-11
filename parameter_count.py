import run
from utils.config import Config
import logging
import sys
from utils import set_config

logging.basicConfig(level=logging.INFO)

model_list = [
    "MGNet18_prog_var",
    "MGNet18_cp_var",
    "MGNet18_ce_var",
    "MGNet18_prog_const",
    "MGNet18_cp_const",
    "MGNet18_ce_const",
    
    "FASMGNet18_1_prog_rfpad_var",
    "FASMGNet18_1_cp_rfpad_var",
    "FASMGNet18_1_ce_rfpad_var",
    "FASMGNet18_1_ce_interpol_var",
    "FASMGNet18_1_cp_interpol_var",
    "FASMGNet18_1_prog_interpol_var",
    "FASMGNet18_1_prog_rfpad_const",
    "FASMGNet18_1_cp_rfpad_const",
    "FASMGNet18_1_ce_rfpad_const",
    "FASMGNet18_1_ce_interpol_const",
    "FASMGNet18_1_cp_interpol_const",
    "FASMGNet18_1_prog_interpol_const",
    
    "FASMGNet18_2_prog_rfpad_var",
    "FASMGNet18_2_cp_rfpad_var",
    "FASMGNet18_2_ce_rfpad_var",
    "FASMGNet18_2_ce_interpol_var",
    "FASMGNet18_2_cp_interpol_var",
    "FASMGNet18_2_prog_interpol_var",
    "FASMGNet18_2_prog_rfpad_const",
    "FASMGNet18_2_cp_rfpad_const",
    "FASMGNet18_2_ce_rfpad_const",
    "FASMGNet18_2_ce_interpol_const",
    "FASMGNet18_2_cp_interpol_const",
    "FASMGNet18_2_prog_interpol_const",
    
    "MGNet34_prog_var",
    "MGNet34_cp_var",
    "MGNet34_ce_var",
    "MGNet34_prog_const",
    "MGNet34_cp_const",
    "MGNet34_ce_const",
    
    "FASMGNet34_1_prog_rfpad_var",
    "FASMGNet34_1_cp_rfpad_var",
    "FASMGNet34_1_ce_rfpad_var",
    "FASMGNet34_1_ce_interpol_var",
    "FASMGNet34_1_cp_interpol_var",
    "FASMGNet34_1_prog_interpol_var",
    "FASMGNet34_1_prog_rfpad_const",
    "FASMGNet34_1_cp_rfpad_const",
    "FASMGNet34_1_ce_rfpad_const",
    "FASMGNet34_1_ce_interpol_const",
    "FASMGNet34_1_cp_interpol_const",
    "FASMGNet34_1_prog_interpol_const",
    
    "FASMGNet34_2_prog_rfpad_var",
    "FASMGNet34_2_cp_rfpad_var",
    "FASMGNet34_2_ce_rfpad_var",
    "FASMGNet34_2_ce_interpol_var",
    "FASMGNet34_2_cp_interpol_var",
    "FASMGNet34_2_prog_interpol_var",
    "FASMGNet34_2_prog_rfpad_const",
    "FASMGNet34_2_cp_rfpad_const",
    "FASMGNet34_2_ce_rfpad_const",
    "FASMGNet34_2_ce_interpol_const",
    "FASMGNet34_2_cp_interpol_const",
    "FASMGNet34_2_prog_interpol_const",

    "DenseNet121",
    "FractalNet3",
    "FractalNet4",
    "ResNet18",
    "ResNet34",
    "MRN_Net18",
    "MRN_Net34",
]

for model in model_list:
    logging.info("\n\n############## NEW MODEL ###################\n\n")
    Config().change_value("Setup", "Architecture", model)
    logging.info("Using Model:\t%s" % (model))
    
    net = set_config.choose_architecture()
    print("Using Model:\t%s" % (model))
    pytorch_total_params = round(sum(p.numel() for p in net.parameters() if p.requires_grad) / 1000000,2)
    print(pytorch_total_params)