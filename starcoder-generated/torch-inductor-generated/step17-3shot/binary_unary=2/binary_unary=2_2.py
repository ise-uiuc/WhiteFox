
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.pool1(x1)
        v2 = v1 - 0.5
        v3 = F.relu(v2)
        v4 = self.pool1(v3)
        v5 = v4 - 2
        v6 = F.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 32, 32, 32)
