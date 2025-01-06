import os, sys, logging
from glob import glob
from collections import defaultdict
import re

import pandas as pd
import numpy as np

import util

logger = logging.getLogger(__name__)


def main(model_names):
    logger.info('model names %s', model_names)
    for model_name in model_names.split():
        if not model_name.startswith('AR'):
            continue
        ckpt_dir = sorted(glob(f"../data/{model_name}/checkpoint-*"), key=lambda x: int(x.split('-')[-1]))[-1]
        fpath = f"{ckpt_dir}/adapter_config.json"
        config = util.load_json(fpath)
        if 'hfmodels' not in config["base_model_name_or_path"]:
            config["base_model_name_or_path"] = f"../data/hfmodels/{config['base_model_name_or_path']}"
            util.dump_json(config, fpath)
        to_fpath =  re.sub("../data/hfmodels", "../data/submission/yaa/data/hfmodels", config["base_model_name_or_path"])
        if not os.path.exists(to_fpath):
            os.system(f"cp -r {config['base_model_name_or_path']} {to_fpath}")

if __name__ == "__main__":
    util.set_logger()
    main(sys.argv[1])
