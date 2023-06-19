import os
import yaml

class CfgNode(object):

    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else: 
            return CfgNode(value) if isinstance(value, dict) else value

    def _merge(self, cfg_node):
        
        for key, value in cfg_node.__dict__.items(): 
            
            if not hasattr(self, key):
                setattr(self, key, value)
            else: 
                assert isinstance(getattr(self, key), type(value)), "Mismatch between 'base config' and 'config' file!"

                if not isinstance(value, CfgNode): 
                    setattr(self, key, value)
                else:
                    setattr(self, key, getattr(self, key)._merge(value))
    
        return self 


def set_config(config_file): 
    
    try:
        config = yaml.safe_load(open(config_file))
    except Exception as e: 
        print(e)
        print('Wrong config file!')
    
    base_config_file = os.path.relpath(os.path.join(os.path.dirname(config_file), config['_BASE_']))
    
    try:
        base_config = yaml.safe_load(open(base_config_file))
    except Exception as e: 
        print(e)
        print('Wrong base config file!')

    return CfgNode(base_config)._merge(CfgNode(config))