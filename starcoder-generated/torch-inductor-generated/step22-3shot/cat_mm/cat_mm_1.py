
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(64, 128, 3, padding=1)
    def forward(self,x1):
        return torch.cat([self.conv1(x1), self.conv1(x1)], 1)
# Inputs to the model
x1 = torch.randn(1, 64, 3, 13, 64)
