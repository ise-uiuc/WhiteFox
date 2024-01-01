
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 1, padding=0)
        self.conv2 = torch.nn.Conv2d(1, 12, 1, padding=0)
    def forward(self, x1):
        v1 = torch.squeeze(self.conv1(x1), 1)
        v2 = torch.squeeze(self.conv2(x1), 1)
        v3 = torch.add(v1, v2)
        v4 = F.relu(v3)

        return v4

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
