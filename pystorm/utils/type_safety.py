from .minitorch import as_tensor, from_numpy, __default_dtype__

def ensure_torch(x, type_float = False):
    try:
        x = as_tensor(x)
        if type_float:
            x = x.type(__default_dtype__)
    except:
        try:
            x = from_numpy(x)
            if type_float:
                x = x.type(__default_dtype__)
        except:
            pass
    
    if type_float:
        try:
            x = x.type(__default_dtype__)
        except:
            pass
    return x

def ensure_numpy(x):
    
    try:
        x = x.detach()
    except:
        pass
    
    try:
        x = x.to('cpu')
    except:
        pass
    
    try:
        x = x.numpy()
    except:
        pass
    
    return x