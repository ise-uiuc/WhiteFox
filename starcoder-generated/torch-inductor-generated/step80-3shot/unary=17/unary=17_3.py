
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = torch.nn.ConvTranspose2d(1, 1, 2, stride=(1, 1), padding=(1, 1))
        self.conv1 = torch.nn.ConvTranspose2d(1, 145, 4, stride=4)
        self.conv2 = torch.nn.ConvTranspose2d(145, 1, 2, stride=(1, 1), padding=(1, 1))
    def forward(self, x):
        v1 = self.conv3(x)
        v2 = torch.relu(v1)
        v4 = self.conv1(v2)
        v5 = torch.relu(v4)
        v6 = self.conv2(v5)
        v7 = torch.max(v6)
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 10, 10)
