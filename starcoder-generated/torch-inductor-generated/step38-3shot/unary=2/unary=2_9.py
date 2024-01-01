
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        return v3 * 0.044715
# Inputs to the model
x1 = torch.zeros(1, 1, 8, 8)
