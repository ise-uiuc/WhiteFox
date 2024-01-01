
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 5)
    # def forward(self, *x1):
    def forward(self, x1):
        v1 = F.relu(self.conv1(x1))
        v2 = F.relu(self.conv2(v1))
        v3 = F.relu(self.conv3(v2))
        v4 = v3.view(v3.size(0), -1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 200, 200)
