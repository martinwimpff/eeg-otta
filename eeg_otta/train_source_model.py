from argparse import ArgumentParser
from datetime import datetime
import os
from pathlib import Path
import yaml

import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from eeg_otta.models import BaseNet
from eeg_otta.utils.get_datamodule_cls import get_datamodule_cls
from eeg_otta.utils.seed import seed_everything

CHECKPOINT_PATH = os.path.join(Path(__file__).resolve().parents[1], "checkpoints")
CONFIG_DIR = os.path.join(Path(__file__).resolve().parents[1], "configs")
DEFAULT_CONFIG = "bcic2a_within_basenet.yaml"


def train_source_model(config):
    # get datamodule_cls and model_cls
    model_cls = BaseNet
    datamodule_cls = get_datamodule_cls(dataset_name=config["dataset_name"])

    if config["subject_ids"] == "all":
        subject_ids = datamodule_cls.all_subject_ids
    else:
        subject_ids = [config["subject_ids"]]
    datamodule = datamodule_cls(config["preprocessing"], subject_ids=subject_ids)

    test_accs = []
    now = datetime.now()
    run_name = f"src-{config['dataset_name']}" + now.strftime("_%Y-%m-%d_%H-%M-%S")

    # save config
    os.makedirs(os.path.join(CHECKPOINT_PATH, run_name), exist_ok=False)
    with open(os.path.join(CHECKPOINT_PATH, run_name, "config.yaml"), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    for subject_id in subject_ids:
        seed_everything(config["seed"])

        # set up the trainer
        checkpoint_cb = ModelCheckpoint(
            dirpath=os.path.join(CHECKPOINT_PATH, run_name, str(subject_id)),
            filename="model")
        trainer = Trainer(
            callbacks=[checkpoint_cb],
            max_epochs=config["max_epochs"],
            logger=False
        )

        # set subject_id
        datamodule.subject_id = subject_id

        # train model
        model = model_cls(**config["model_kwargs"], max_epochs=config["max_epochs"])
        trainer.fit(model, datamodule=datamodule)

        # test model
        test_results = trainer.test(model, datamodule)
        test_accs.append(test_results[0]["test_acc"])

    print(f"source accuracy: {100 *np.mean(test_accs):.2f}%")


if __name__ == "__main__":
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    # load config
    with open(os.path.join(CONFIG_DIR, args.config)) as f:
        config = yaml.safe_load(f)

    train_source_model(config)
