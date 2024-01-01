
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()
    def forward(self, x5):
        v1 = torch.tanh(x5)
        return v1
# Inputs to the model
x5 = torch.randn(1, 5, 3, 3)
