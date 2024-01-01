
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
    def forward(self, x4):
        v1 = self.conv(x4)
        v2 = torch.tanh(v1)
        return v1 + v2
# Inputs to the model
tensor = torch.randn(1, 2, 3, 3)
