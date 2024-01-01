
from torch import nn

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        t1 = x1.view(1, 36, -1)
        t2 = torch.matmul(t1, t1.transpose(-2, -1).float())
        t3 = torch.div(t2.transpose(-2, -1), 8192)
        t4 = t3.softmax(dim=-1)
        t5 = nn.functional.dropout(t4, p=1)
        ans = t5.matmul(t1)
        return ans

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 256)
