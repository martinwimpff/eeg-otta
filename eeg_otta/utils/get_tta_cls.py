from eeg_otta.tta import Norm, OnlineAlignment, EntropyMinimization


def get_tta_cls(tta_method: str):
    if tta_method == "alignment":
        return OnlineAlignment
    elif tta_method == "norm":
        return Norm
    elif tta_method == "entropy_minimization":
        return EntropyMinimization
    else:
        raise NotImplementedError
