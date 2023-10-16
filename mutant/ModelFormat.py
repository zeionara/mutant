from enum import Enum


class ModelFormat(Enum):
    ST = 'safetensors'
    PT = 'torch'
    TF = 'tensorflow'

    @classmethod
    def from_path(cls, path: str):
        if path.endswith('h5'):
            return cls.TF

        if path.endswith('bin'):
            return cls.PT

        if path.endswith('st') or path.endswith('safetensors'):
            return cls.ST

        raise ValueError(f'Cannot infer model format from path {path}')
