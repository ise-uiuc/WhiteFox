
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 8, 1)
        self.dropout2d = torch.nn.Dropout2d(p=0.5)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.dropout2d(v1)
        v3 = v2 - 6.0
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
