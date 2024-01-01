
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(6, 8, (3, 5), stride=(2, 2))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 1.4
        v3 = F.relu(v2)
        v4 = torch.squeeze(v3, 0)
        return v4
# Inputs to the model
x1 = torch.randn(1, 6, 224, 224)
