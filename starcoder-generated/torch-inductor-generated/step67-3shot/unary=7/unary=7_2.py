
```
import torch.nn as nn
class Test(nn.Module):
    def __init__(self, in_, out_):
        super().__init__()
        self.fc = nn.Linear(in_, out_)
 
    def forward(self, x1):
        fc_out = self.fc(x1)
        fc_clamp = torch.clamp(fc_out, min=0, max=6)
        fc_add = fc_out + 3
        fc_mul = fc_clamp * fc_add
        return fc_mul / 6
```

# Initializing the model
m = Test(100, 10)

# Inputs to the model
x1 = torch.randn(1, 100)
