
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = torch.nn.ReLU6()
        self.conv1 = torch.nn.ConvTranspose2d(3, 16, (7, 7), stride=(2, 2))
        self.conv2 = torch.nn.ConvTranspose2d(16, 4, (1, 1), stride=(1, 1), padding=(0, 0))
    def forward(self, x1):
        v1 = self.relu1(x1)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 18, 18)
