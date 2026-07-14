import random
import os
import numpy as np
import torch

def setup_seed(seed=999):
    # 1. Set seed for built-in Python pseudo-random generator
    random.seed(seed)
    
    # 2. Set seed for Python's hash function (important for data structures)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. Set seed for NumPy operations
    np.random.seed(seed)
    
    # 4. Set seed for PyTorch CPU operations
    torch.manual_seed(seed)
    
    # 5. Set seed for PyTorch GPU operations (single and multi-GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # 6. Force PyTorch to use deterministic algorithms (ensures identical outputs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
