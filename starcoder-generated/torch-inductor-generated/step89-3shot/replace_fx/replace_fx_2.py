
from torch.fx.experimental.optimization.random_patterns import lowmem_dropout, rand_like

class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs, weights):
        # t0 = torch.nn.functional.dropout(inputs, p=0.5)
        t1 = lowmem_dropout(inputs, p=0.5)
        t2 = rand_like(inputs)
        t3 = rand_like(inputs)
        return torch.cat((t3, t2, t1), 0)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
w1 = torch.randn(3, 2, 2)
