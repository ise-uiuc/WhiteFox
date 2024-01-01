
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 1, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.relu(v1)
        v3 = self.conv_transpose(v2)
        return v3
# Input to model
x = torch.randn(1, 1, 8, 8)
