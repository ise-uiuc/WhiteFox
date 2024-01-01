
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(1, 4, 1)
    def forward(self, x):
        v1 = torch.tanh(self.conv3d(x))
        return v1
# Inputs to the model
x = torch.randn(1, 1, 1, 5, 5)
