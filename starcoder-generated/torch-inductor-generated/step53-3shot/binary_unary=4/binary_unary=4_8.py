
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 16, bias=False)
 
    def forward(self, x1, other=None):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
__keyword-arguments__ = {}
# other can be any valid PyTorch tensor
__keyword-arguments__["other"] = __other__
