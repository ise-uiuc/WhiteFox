
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Conv layer
        self.conv1 = torch.nn.ConvTranspose2d(4, 4, 1, stride = 2)
        # Batch Norm layer
        self.bn1 = torch.nn.BatchNorm2d(4)
    def forward(self, x1):
        # Conv Layer
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        # Batch Norm Layer
        v3 = self.bn1(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 16, 16)
