
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        n1 = self.tanh(x)
        x1 = torch.abs(n1)
        return n1
# Inputs to the model
x = torch.randn(1, 1, 64, 64)
