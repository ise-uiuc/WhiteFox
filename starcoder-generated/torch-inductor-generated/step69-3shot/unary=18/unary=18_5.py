
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.relu(v1)
        v3 = self.flatten(v2)
        v4 = torch.nn.Dropout(0.3)(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
