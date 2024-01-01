
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 16, 3, stride=2)
        self.max_pool = torch.nn.MaxPool2d(3, 2, padding=0)
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 32, 3, padding=1, stride=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.max_pool(v2)
        v4 = self.conv_transpose(v3)
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
