
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, (1, 1), stride=(1, 1))
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = torch.relu(x1)
        x4 = x1 + x2
        x5 = torch.cat((x3, x4), dim=-1)
        x6 = torch.sigmoid(x5)
        return x6
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
