
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 7, 1, stride=1, padding=1)
    def forward(self, input, weight=None):
        v1 = self.conv(input)
        if weight == None:
            weight = v1.shape
        v2 = v1 + weight
        return v2
# Inputs to the model
input = torch.randn(1, 1, 64, 64)
