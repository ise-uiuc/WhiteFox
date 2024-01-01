
import math
class Model(torch.nn.Module):
    def __init__(self, dropout_p: float):
        super().__init__()
        self.dropout_p = dropout_p
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(self.dropout_p)
 
    def forward(self, x1, x2):
        v1 = x1.matmul(x2.transpose(-2, -1))
        scale_factor = math.sqrt(x2.size(-1))
        inv_scale_factor = 1.0 / scale_factor
        v2 = v1.div(inv_scale_factor)
        v3 = self.softmax(v2)
        v4 = self.dropout(v3)
        v5 = v4.matmul(x2)
        return v5

# Initializing the model
m = Model(0.2)

# Inputs to the model
x1 = torch.randn(1, 32, 6)
x2 = torch.randn(1, 6, 4)
