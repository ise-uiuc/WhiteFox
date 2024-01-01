
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.nn.functional.relu6(v1)
        v3 = 3 + v2
        v4 = v3.clamp_(min=0, max=6)
        v5 = v4.div(6)
        v6 = self.conv2(v5)
        v7 = torch.nn.functional.relu6(v6)
        v8 = 3 + v7
        v9 = v8.clamp_(min=0, max=6)
        v10 = v9.div(6)
        return v10
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
