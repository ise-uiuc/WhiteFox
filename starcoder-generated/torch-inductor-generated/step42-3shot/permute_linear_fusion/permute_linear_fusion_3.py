
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1, 3)
        v2 = torch.nn.functional.conv2d(v1, self.conv.weight, None, [1, 1], [1, 0], [1, 0], 1, False)
        v2 = torch.nn.functional.hardtanh(v1, -1, 6)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
