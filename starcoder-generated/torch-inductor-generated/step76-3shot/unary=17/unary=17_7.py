
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 4, (12, 9), stride=(1, 1), bias=True)
        self.conv2 = torch.nn.ConvTranspose2d(4, 8, (9, 4), stride=(1, 1), bias=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 40, 20)
