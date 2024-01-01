
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(3, 32, 3, padding=1, stride=2)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(32, 16, 2, padding=0, stride=1)
        self.max_pool = torch.nn.MaxPool2d(2, stride=1)
        self.avg_pool = torch.nn.AvgPool2d(2, stride=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = self.relu(v1)
        v3 = self.conv_transpose_2(v2)
        v4 = self.relu(v3)
        v5 = self.max_pool(v4)
        v6 = self.avg_pool(v5)
        v7 = torch.add(v4, v6)
        v8 = torch.add(v7, v1)
        v9 = torch.sigmoid(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 128, 32)
