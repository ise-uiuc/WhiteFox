
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(2, 2), stride=(2, 2), padding=0, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(2, 2), stride=(1, 1), padding=0, bias=False)
        self.conv3 = nn.Conv2d(32, 28, kernel_size=(2, 2), stride=(2, 2), padding=0, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        v3 = self.conv2(v2)
        v4 = F.relu(v3)
        v5 = self.conv3(v2)
        v6 = F.relu(v5)
        v7 = torch.tanh(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
