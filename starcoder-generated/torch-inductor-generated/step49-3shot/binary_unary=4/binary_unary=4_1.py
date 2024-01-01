
from collections import OrderedDict
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
        self.linear_add_weight = torch.nn.Parameter(torch.rand(4, 4))
 
    def forward(self, x1, _linear_add_weight=None):
        v1 = self.linear(x1)
        if _linear_add_weight is not None:
            v1 = v1 + _linear_add_weight
        v2 = F.relu(v1)
        return v2
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
kw1 = OrderedDict([('_linear_add_weight', torch.tensor([[0.1, 0.2, 0.3, 0.4], [1.1, 1.2, 1.3, 1.4], [2.1, 2.2, 2.3, 2.4], [3.1, 3.2, 3.3, 3.4]]))])
