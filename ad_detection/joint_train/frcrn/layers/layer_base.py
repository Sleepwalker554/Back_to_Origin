import abc

import torch.nn as nn


class LayerBase(nn.Module):

    def __init__(self):
        super(LayerBase, self).__init__()

    @abc.abstractmethod
    def to_kaldi_nnet(self):
        pass
