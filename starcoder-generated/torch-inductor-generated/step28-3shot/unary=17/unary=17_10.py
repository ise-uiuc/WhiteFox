
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 5, 3, padding=(1, 1, 1, 1), stride=(1, 1))
        self.conv2 = torch.nn.ConvTranspose2d(5, 5, 3, padding=(1, 1, 1, 1), stride=(1, 1))
        self.conv3 = torch.nn.ConvTranspose2d(5, 3, 3, padding=(1, 1, 1, 1), stride=(1, 1))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        v6 = torch.relu(v5)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
