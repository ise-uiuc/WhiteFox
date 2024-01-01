
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias1 = torch.nn.Parameter(torch.Tensor(1))
        self.bias2 = torch.nn.Parameter(torch.Tensor(1))
 
    def forward(self, input1):
        v1 = torch.nn.functional.linear(input1, torch.ones(256), self.bias1)
        v2 = v1.clamp(min=0, max=6)
        v3 = v2 + 3
        v4 = v3 * 6
        return v4

# Initializing the model
m = Model()

# Inputs to the model
__input__ = torch.randn(1, 256)
