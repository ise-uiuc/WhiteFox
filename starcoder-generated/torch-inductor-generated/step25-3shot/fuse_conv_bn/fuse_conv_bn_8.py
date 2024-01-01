
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=(1, 1), stride=(2, 2))
        self.bn = torch.nn.BatchNorm2d(num_features=1, affine=False)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) # this works even though self.conv1.bias is not None
        return x
# Inputs to the model
x = torch.randn(1, 1, 2, 2)
