
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 7, stride=2, padding=3)
    def forward(self, x):
        x = self.conv1(x)
        return x
# Inputs to the model
x = torch.rand(1, 1, 224, 224)
