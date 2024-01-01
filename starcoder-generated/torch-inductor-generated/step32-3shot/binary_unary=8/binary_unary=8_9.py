
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 3)
    def forward(self, x1):
        conv_weight = self.conv.weight
        v1 = torch.nn.functional.conv2d(x1, conv_weight, padding=1, stride=2)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
