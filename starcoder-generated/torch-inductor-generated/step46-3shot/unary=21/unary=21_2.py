
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        tanh_v = self.tanh(x)
        return tanh_v
# Inputs to the model
x = torch.randn(1, 1, 6, 6)
