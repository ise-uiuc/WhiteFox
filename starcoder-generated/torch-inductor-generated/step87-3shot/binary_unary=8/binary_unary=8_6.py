
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv10 = torch.nn.Conv2d(3, 8, 3, stride=2, padding=0)
        self.conv11 = torch.nn.Conv2d(3, 8, 3, stride=2, padding=0)
    def forward(self, x1):
        v8 = self.conv10(x1)
        v9 = self.conv11(x1)
        v10 = v8 + v9
        v11 = torch.relu(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
