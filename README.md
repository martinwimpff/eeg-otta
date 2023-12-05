# Calibration-free online test-time adaptation for electroencephalography motor imagery decoding

This is the official repository to the paper [Calibration-free online test-time adaptation for electroencephalography motor imagery decoding](https://arxiv.org/abs/2311.18520). The implementation is based on [mariodoebler/test-time-adaptation](https://github.com/mariodoebler/test-time-adaptation). Additionally we use [BaseNet](https://arxiv.org/abs/2310.11198) from this [repository](https://github.com/martinwimpff/channel-attention).

## Usage
### Installation
- clone this repository
- run `pip install .` to install the `eeg-otta` package

_Note: you can also use poetry for the installation_
### Source training
- run [train_source_model.py](eeg_otta/train_source_model.py) with the `--config` of your choice, the checkpoints and the config will automatically saved in the [checkpoints directory](checkpoints)

_Note: you can also use one of the checkpoints in the [checkpoints directory](checkpoints)_
### Run the online test-time adaptation
- run [run_adaptation.py](eeg_otta/run_adaptation.py) with the `--config` of your choice (one of the [configs](configs) starting with `tta`)

_Note: check the `source_run` parameter in the yaml file_

## Citation
If you find this repository useful, please cite our work
```
@article{wimpff2023calibration,
  title={Calibration-free online test-time adaptation for electroencephalography motor imagery decoding},
  author={Wimpff, Martin and D{\"o}bler, Mario and Yang, Bin},
  journal={arXiv preprint arXiv:2311.18520},
  year={2023}
}
```