
class Model(torch.nn.Module):
    def __init__(self):
       super(resnet18, self).__init__()
       self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    def forward(self, x):
        v1 = self.conv1(x)
        return x + 3
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
