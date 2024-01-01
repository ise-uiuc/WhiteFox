
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 8, 1, stride=2)
        self.conv2 = torch.nn.Conv2d(112, 20, 3, stride=2)
        self.conv3 = torch.nn.Conv2d(40, 32, 5, stride=1)
    def forward(self, x1, x2):
        v1 = torch.cat((x1, x2), 1) # Connect the tensors with dimension 1, i.e., channel dimension
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        v5 = torch.tanh(v4)
        return v5
# Inputs to the model
x1 = torch.randn(64,8,32,32)
x2 = torch.randn(64,112,24,24)
