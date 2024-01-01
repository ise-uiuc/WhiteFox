
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(1, 1, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = x1.view(-1, 10)
        v3 = torch.matmul(v2, v1)
        v4 = torch.relu(v3)
        v5 = v4.view(-1, 1, 4, 4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 25)
