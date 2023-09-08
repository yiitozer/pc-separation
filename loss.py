import torch


class Loss:
    def __init__(self,
                 type: str):
        self._type = type

    def process(self, x, y):
        if self._type == 'mse':
            return torch.nn.functional.mse_loss(x, y)
