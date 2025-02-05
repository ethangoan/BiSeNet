

from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2, BayesBiSeNetV2
from .enetv2 import ENet
from .pidnet import PIDNet
from .ppliteseg import PPLiteSegT, PPLiteSegB


model_factory = {
    'bisenetv1': BiSeNetV1,
    'bisenetv2': BiSeNetV2,
    'bayes_bisenetv2': BayesBiSeNetV2,
    'enet': ENet,
    'pidnet': PIDNet,
    'bayes_pidnet': PIDNet,
    'ppliteseg': PPLiteSegT,
    'pplitesegb': PPLiteSegB
}
