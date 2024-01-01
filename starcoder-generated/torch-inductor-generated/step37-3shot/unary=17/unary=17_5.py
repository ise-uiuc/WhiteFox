
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 4, 3, padding=1, stride=2)
        self.max_pool = torch.nn.MaxPool2d(2, 1, padding=1)
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.max_pool(x1)
        v4 = self.tanh(v2)
        v5 = self.sigmoid(v2)
        return v4, v5
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
