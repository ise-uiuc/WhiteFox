
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 5, 3, stride=1, padding=1)
        self.max_pool = torch.nn.MaxPool2d(3, 1, padding=1)
        self.conv = torch.nn.Conv2d(5, 8, 3, stride=1, padding=1)
        self.batch_normalization = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.max_pool(v1)
        v3 = torch.relu(v2)
        v5 = self.conv(v3)
        v6 = self.batch_normalization(v5)
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
