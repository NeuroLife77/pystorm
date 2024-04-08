from .minitorch import as_tensor, from_numpy

def ensure_torch(x, type_float = False):
    try:
        x = as_tensor(x)
        if type_float:
            x = x.float()
    except:
        try:
            x = from_numpy(x)
            if type_float:
                x = x.float()
        except:
            pass
    
    if type_float:
        try:
            x = x.float()
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