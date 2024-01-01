
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, add):
        v1 = self.conv1(x1)
        if add:
            v2 = self.conv2(x1)
            v3 = v1 + v2
        else:
            v3 = v1 + x1
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        if add:
            v7 = v5 + v5
        else:
            v7 = v5
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
add = True
