
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 256, 7, stride=2, padding=3)
        torch.nn.init.kaiming_uniform(self.conv1.weight, mode='fan_in')
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - -8
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
