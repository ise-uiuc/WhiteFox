
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = x.mean()
        v2 = self.tanh(v1)
        return v2
# Inputs to the model
tensor = torch.randn(1, 3, 224, 224)
