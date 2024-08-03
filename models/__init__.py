import numpy as np

np.random.seed(0)

from models.cls_models.LEPDNet import LEPDNet

MODELS = {
    'LEPDNet': LEPDNet,
}
