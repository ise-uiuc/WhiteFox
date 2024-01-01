
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(28, 32)
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 64, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_transpose(v2)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 28)
