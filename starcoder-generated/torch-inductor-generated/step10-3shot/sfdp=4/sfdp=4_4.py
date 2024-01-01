
import math

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Parameter(torch.randn(50, 8, dtype=torch.float), requires_grad=True)
        self.k = torch.nn.Parameter(torch.randn(3, 2, dtype=torch.float), requires_grad=True)
        self.v = torch.nn.Parameter(torch.randn(4, 2, dtype=torch.float), requires_grad=True)
 
    def forward(self, x1, x2):
        qk = self.q @ self.k.transpose(-2, -1) / math.sqrt(self.q.size(-1))
        qk = qk + x2 # Attention mask is passed by applying a residual to the output of the key dot product
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ self.v
        return qk, attn_weight, output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 50, 8)
x2 = torch.randn(1, 3, 2)
__output__, __state1__, __state2__ = m(x1, x2)

