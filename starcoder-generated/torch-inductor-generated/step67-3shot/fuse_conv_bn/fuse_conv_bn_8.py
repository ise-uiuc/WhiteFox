
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 512, kernel_size=11, stride=4, padding=2)
        self.conv2 = torch.nn.Conv2d(512, 512, kernel_size=512, stride=4, padding=2)
    def forward(self, x):
        x1  = self.conv1(x)
        return self.conv2(x1)
# Inputs to the model
x = torch.randn(1, 1, 224, 224)
