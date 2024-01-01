
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 11, stride=1, padding=5)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.nn.functional.batch_norm(v1, x.shape[1], eps=0.0001)
        v3 = self.relu(v2)
        v4 = self.conv(v3)
        # v5 = torch.nn.MaxPool2d(3, stride=1, padding=1)(v4)
        v5 = torch.squeeze(v4)
        # v6 = torch.nn.functional.interpolate(v5, size=7, mode='nearest')
        v6 = v5.reshape(1, 8*8*16)
        v7 = torch.nn.functional.linear(v6, v6)
        v8 = torch.squeeze(v7)
        v9 = self.relu(v8)
        return v9
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
