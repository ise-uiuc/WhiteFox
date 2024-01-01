
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.5
        v3 = F.relu(v2)
        v4 = torch.reshape(v3, (1, 16, 16*16))
        v5 = F.max_pool2d(v4)
        v6 = torch.squeeze(v5, dim=0)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
