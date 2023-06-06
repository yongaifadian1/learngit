from .config import _C as Cfg
import json

#命令行传参修改Cfg对象
def modify_config(cfg, args):
    # 命令行传参args是namespace类型
    args = vars(args)  # Namespace -> Dict 转字典
    args = dict_2_list(args)#字典转列表
    print('Use arguments from command line:', args)
    cfg.merge_from_list(args)#将命令行参数合并到Cfg对象中

    modeltype = cfg.model.type
    database = cfg.dataset.database
    feature = cfg.dataset.feature
    #用字典读取json
    train_config = json.load(open(f'./config/train_{modeltype}.json', 'r'))[database][feature]
    model_config = json.load(open('./config/model_config.json', 'r'))[modeltype]

    # 从json读取训练配置和模型配置，合并到Cfg对象
    train_config = dict_2_list(train_config)
    print('Use arguments from train json file:', train_config)
    cfg.merge_from_list(train_config)
    
    # modify cfg.mark  #标识不同的下游模型
    if modeltype == 'Transformer':
        num_layers = model_config['num_layers']
        _mark = f'L{num_layers}'

    else:
        raise ValueError(f'Unknown model type: {modeltype}')

    cfg.mark = _mark + '_' + cfg.mark if cfg.mark is not None else _mark
    print('Modified mark:', cfg.mark)
    
    # add key: evaluate 训练的评估方式
    cfg.train.evaluate = json.load(open(f'./config/{database}_feature_config.json', 'r'))['evaluate']

def dict_2_list(dict):
    lst = []
    for key, value in dict.items():
        if value is not None:
            lst.extend([key, value])
    return lst
