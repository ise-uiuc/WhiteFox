
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1, dilation=2)
        self.softmax = torch.nn.Softmax2d()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        v3 = torch.squeeze(v2, 0)
        v4 = v3.reshape(1,-1)
        v5 = self.softmax(v4)
        v6 = v4 * v5
        v7 = torch.reshape(v6, (1, 8, 30, 30))
        v8 = torch.squeeze(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
