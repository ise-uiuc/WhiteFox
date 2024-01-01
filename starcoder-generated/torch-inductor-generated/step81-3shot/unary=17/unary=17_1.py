
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 10, (2, 3), padding=(0, 0), stride=(2, 1), bias=False)
        self.conv2 = torch.nn.ConvTranspose2d(10, 5, 4, padding=(1, 1), stride=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        v5 = torch.nn.functional.interpolate(v4, scale_factor=2, mode='nearest')
        return v5
# Inputs to the model
x1 = torch.randn(1, 4, 22, 25)
