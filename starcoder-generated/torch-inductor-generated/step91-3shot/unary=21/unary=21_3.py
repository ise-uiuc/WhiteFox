
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()
    def forward(self, x3):
        v1 = self.tanh(x3)
        v2 = self.tanh(v1)
        return v2 + x3
# Inputs to the model
tensor = torch.randn(1, 16, 2, 2)
