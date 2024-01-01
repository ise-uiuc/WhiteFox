
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, 2, stride=2, padding=0)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(3072, 2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        v3 = torch.transpose(v2, 1, 0)
        v4 = self.flatten(v3)
        v5 = self.linear(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 1024, 64)
