from argparse import ArgumentParser
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from models import BaseNet
from utils.get_accuracy import get_accuracy
from utils.get_datamodule_cls import get_datamodule_cls
from utils.get_tta_cls import get_tta_cls
from utils.seed import seed_everything

CHECKPOINT_PATH = os.path.join(Path(__file__).resolve().parents[1], "checkpoints")
CONFIG_DIR = os.path.join(Path(__file__).resolve().parents[1], "configs")
DEFAULT_CONFIG = "tta_alignment.yaml"


def run_adaptation(config):
    # load source config
    with open(os.path.join(CHECKPOINT_PATH, config["source_run"], "config.yaml")) as f:
        source_config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datamodule_cls = get_datamodule_cls(source_config["dataset_name"])
    model_cls = BaseNet
    tta_cls = get_tta_cls(config["tta_method"])

    if source_config["subject_ids"] == "all":
        subject_ids = datamodule_cls.all_subject_ids
    else:
        subject_ids = [source_config["subject_ids"]]

    source_config["preprocessing"]["alignment"] = False
    source_config["preprocessing"]["batch_size"] = 1
    datamodule = datamodule_cls(source_config["preprocessing"], subject_ids=subject_ids)

    test_accs = []
    for version, subject_id in enumerate(subject_ids):
        seed_everything(source_config["seed"])

        # load checkpoint
        ckpt_path = os.path.join(CHECKPOINT_PATH, config["source_run"], str(subject_id),
                                 "model.ckpt")
        model = model_cls.load_from_checkpoint(ckpt_path, map_location=device)

        # set subject_id
        datamodule.subject_id = subject_id
        datamodule.prepare_data()
        datamodule.setup()

        model = tta_cls(model, config["tta_config"], datamodule.info)

        acc = get_accuracy(model, datamodule.test_dataloader(), device)
        test_accs.append(acc)

    # print overall test accuracy
    print(f"test_acc: {100 * np.mean(test_accs):.2f}")


if __name__ == "__main__":
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    # load config
    with open(os.path.join(CONFIG_DIR, args.config)) as f:
        config = yaml.safe_load(f)

    run_adaptation(config)
