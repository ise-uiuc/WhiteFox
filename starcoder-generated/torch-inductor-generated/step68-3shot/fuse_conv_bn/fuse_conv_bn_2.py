
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(4)
    def forward(self, x):
        x = F.relu(self.bn1(x))
        x = self.bn2(x)
        return x
# Inputs to the model
x = torch.randn(1, 4, 28, 28)
