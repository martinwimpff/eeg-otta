from eeg_otta.datamodules import BCICIV2aWithinSubject, BCICIV2aLOSO, \
    BCICIV2bWithinSubject, BCICIV2bLOSO


def get_datamodule_cls(dataset_name: str):
    if dataset_name == "bcic2a_within":
        return BCICIV2aWithinSubject
    elif dataset_name == "bcic2a_loso":
        return BCICIV2aLOSO
    elif dataset_name == "bcic2b_within":
        return BCICIV2bWithinSubject
    elif dataset_name == "bcic2b_loso":
        return BCICIV2bLOSO
    else:
        raise NotImplementedError
