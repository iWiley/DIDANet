from enum import Enum, auto

class ModelType(Enum):
    LASSO = auto()
    IPCRidge = auto()
    Elastic = auto()
    RSF = auto()
    ERSF = auto()

    HL_SSVM = auto()
    F_SSVM = auto()
    N_SSVM = auto()
    FK_SSVM = auto()
    MINLIP = auto()

    GBSA = auto()
    CGBSA = auto()

    DeepSurv = auto()
    DeepHit = auto()
    N_MTLR = auto()

    @staticmethod
    def GetType(type):
        if type in [
            ModelType.LASSO,

            ModelType.HL_SSVM,
            ModelType.F_SSVM,
            ModelType.N_SSVM,
            ModelType.FK_SSVM,
            ModelType.MINLIP,

            ModelType.Elastic,
            ModelType.RSF,
            ModelType.ERSF,
            ModelType.IPCRidge,
            ModelType.GBSA,
            ModelType.CGBSA,
        ]:
            return "ML"
        return "DL"