# flake8: noqa
from .unet import UNet
from .hed_unet import HEDUNet
from .hed import HED
from .boring_backbone import BoringBackbone
from .ocr import OCRNet
from .logistic_regression import LogisticRegression
from .mixture_unet import MixtureUNet
from .segnet import SegNet
from .fcn import FCN32s as FCN
from .liu_jezek import LiuJezek
from .schmittetal import Schmittetal
from .GSCNN.network.gscnn import GSCNN
from .deep_structure_unet import DeepStructureUNet
from .lee import Lee
from .hrnet_ocr.lib.models.seg_hrnet_ocr import HighResolutionNet as HRNet_OCR
from .deepunet import DeepUNet
from .active_contour import ActiveContour
from .dexined import DexiNed

def get_model(model_name):
    try:
        return globals()[model_name]
    except KeyError:
        raise ValueError(f'Can\'t provide Model called "{model_name}"')
