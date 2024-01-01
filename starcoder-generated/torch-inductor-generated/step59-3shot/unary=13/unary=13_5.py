
import math

class Model(torch.nn.Module):
    def __init__(self, linear_transformation):
        super().__init__()
        self.linear_transformation = linear_transformation
 
    def forward(self, x1):
        v1 = self.linear_transformation(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing a model with one linear transformation
def init_model(randomize_weights = False):
    if randomize_weights:
        torch.manual_seed(0)
    L1 = 16
    m = Model(torch.nn.Linear(10, L1))
    if randomize_weights:
        for param in m.parameters():
            param.data = torch.randn_like(param) * math.sqrt(2 / L1)
    return m

# Initializing a model whose input tensor needs to be generated
def torch_erf_model(randomize_weights = False):
    if randomize_weights:
        torch.manual_seed(0)
    m = Model(torch.torch._C._nn.functional.erf)
    if randomize_weights:
        for param in m.parameters():
            param.data = torch.randn_like(param) * math.sqrt(2)
    return m

# Initializing a model whose input tensor needs to be generated
def erf_model(randomize_weights = False):
    if randomize_weights:
        torch.manual_seed(0)
    m = Model(torch.erf)
    if randomize_weights:
        for param in m.parameters():
            param.data = torch.randn_like(param) * math.sqrt(2)
    return m

# Randomly initialize input tensors
x1 = torch.randn(1, 10)
