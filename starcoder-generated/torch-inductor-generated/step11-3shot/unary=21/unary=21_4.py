
class tanhActivation(torch.nn.Module):
    def forward(self, x):
        result = torch.tanh(x)
        return result
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = tanhActivation()
    def forward(self, x):
        v1 = torch.tanh(x)
        v2 = self.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(960, 8)
