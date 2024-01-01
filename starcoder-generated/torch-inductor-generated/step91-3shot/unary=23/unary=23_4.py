
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Create random parameters
        self.param1 = torch.nn.Parameter(torch.randn(32, 43))
        self.param2 = torch.nn.Parameter(torch.randn(23, 12))
    def forward(self, x1):
        v1 = torch.pow(x1, self.param1)
        v2 = torch.softmax(v1, self.param2)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(42, 100000)
