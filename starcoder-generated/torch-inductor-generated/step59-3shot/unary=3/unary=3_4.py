
class ModelWithPadding(torch.nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=2, padding=padding)
    def forward(self, x1):
        v1 = self.conv1(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 56, 56)
