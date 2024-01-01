
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = v1[:,0:-1,:]
        return v2
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
