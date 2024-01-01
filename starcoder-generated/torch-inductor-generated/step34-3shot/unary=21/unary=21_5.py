
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x = torch.zeros(1, 4, 64, 64)
