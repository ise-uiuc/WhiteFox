
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 3, 3)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        s = self.conv(x1)
        return self.relu(s)
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4, 4)
