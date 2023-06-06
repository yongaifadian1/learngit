
from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)
_C.train = CN(new_allowed=True)
_C.model = CN(new_allowed=True)
_C.dataset = CN(new_allowed=True)

_C.train.device = 'cuda'

_C.model.type = 'Transformer'   

# Total epochs for training
_C.train.EPOCH = 120
# The size of a mini-batch 
_C.train.batch_size = 32
# Initial learning rate
_C.train.lr = 0.0005

# Select the GPUs used
_C.train.device_id = "4"


# Select a database to train a model
_C.dataset.database = 'iemocap'   
# Select a kind of feature to train a model
_C.dataset.feature = 'wavlm'    

_C.mark = None#标识不同的下游模型

