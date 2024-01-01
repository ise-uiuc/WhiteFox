
class FusedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.relu = torch.nn.ReLU()
    def forward(self, x4):
        x4 = self.conv(x4)
        return self.relu(x4)
# Inputs to the model
x4 = torch.randn(1, 3, 4, 4)
