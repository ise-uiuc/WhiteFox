
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5,3,1,stride=1,padding=0)
        self.relu = torch.nn.ReLU()
        self.conv_transpose = torch.nn.ConvTranspose2d(4,2,2,stride=1,padding=1)
    def forward(self, x1):
        v1 = torch.nn.functional.interpolate(x1, size=[1, 1])
        v2 = self.relu(self.conv(v1))
        v3 = v2
        v4 = v3
        v5 = self.conv_transpose(v4)
        v6 = self.relu(v5)
        v7 = self.conv_transpose(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 5, 3, 3)
