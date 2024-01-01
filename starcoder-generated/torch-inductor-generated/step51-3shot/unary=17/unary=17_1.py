
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 12, stride=3)
        self.flatten = torch.flatten
        self.linear = torch.nn.Linear(2, 10)
        self.conv_transpose = torch.nn.ConvTranspose2d(12, 24, 3, stride=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = torch.flatten(v2)
        v4 = self.linear(v3)
        v5 = torch.relu(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv_transpose(v6)
        v8 = torch.tanh(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
