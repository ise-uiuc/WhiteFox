
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - torch.tensor([[1, -1, -1], [-1, 2, -1], [0.5, 0.5, -1]])
        return v2
# Inputs to the model
x = torch.randn(1, 3, 83, 83) # 83 = 3 * 83 * 83 + 1 since padding=1
