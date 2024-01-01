
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = torch.nn.ConstantPad2d([12, 12, 12, 12], 1)
    def forward(self, x):
        x = self.pad(x)
        return torch.tanh(x)
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
