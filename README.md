# Calibration-free online test-time adaptation for electroencephalography motor imagery decoding

This is the official repository to the paper [Calibration-free online test-time adaptation for electroencephalography motor imagery decoding](https://ieeexplore.ieee.org/abstract/document/10480468). The implementation is based on [mariodoebler/test-time-adaptation](https://github.com/mariodoebler/test-time-adaptation). Additionally we use [BaseNet](https://iopscience.iop.org/article/10.1088/1741-2552/ad48b9/meta) from this [repository](https://github.com/martinwimpff/channel-attention).

## Usage
### Installation
- clone this repository
- run `pip install .` to install the `eeg-otta` package

_Note: you can also use poetry for the installation_
### Source training
- run [train_source_model.py](eeg_otta/train_source_model.py) with the `--config` of your choice, the checkpoints and the config will automatically saved in the [checkpoints directory](checkpoints)

_Note: you can also use one of the checkpoints in the [checkpoints directory](checkpoints)_
### Run the online test-time adaptation
- run [run_adaptation.py](eeg_otta/run_adaptation.py) with the `--config` and `source_run` of your choice (one of the [configs](configs) starting with `tta`)
- the setting (cross-session or cross-subject/ cross-subject continual) is dependent on your checkpoint i.e. 
whether the within-subject dataset (`_within`) or the leave-one-subject-out (`_loso`) dataset was used. 
- To choose between the cross-subject and cross-subject continual setting, modify the `continual` parameter in the TTA config file (cross-subject is the default).


## Citation
If you find this repository useful, please cite our work
```
@inproceedings{wimpff2024calibration,
  title={Calibration-free online test-time adaptation for electroencephalography motor imagery decoding},
  author={Wimpff, Martin and D{\"o}bler, Mario and Yang, Bin},
  booktitle={2024 12th International Winter Conference on Brain-Computer Interface (BCI)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
```
