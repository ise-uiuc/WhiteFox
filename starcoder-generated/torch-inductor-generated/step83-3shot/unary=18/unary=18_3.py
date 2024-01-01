
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 48, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 2, 5, stride=3, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        #v4 = torch.sigmoid(v3)
        v5 = torch.sigmoid(v3)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
