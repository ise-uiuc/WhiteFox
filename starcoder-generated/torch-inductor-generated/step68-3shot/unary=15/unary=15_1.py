
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 6, 2, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(6, 3, 2, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = torch.reshape(v4, (-1,))
        return v5
# Inputs to the model
x = torch.randn(1,3,64,64)
