
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(False)
    def forward(self, x):
        t = x.type(torch.int16)
        u = self.relu(t)
        return u
# Inputs to the model
tensor = torch.randn(1, 1, 64, 64)
