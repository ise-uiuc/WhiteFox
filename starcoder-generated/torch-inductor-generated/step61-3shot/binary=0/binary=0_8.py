
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 5, 1, stride=1, padding=1)
    def forward(self, input, padding=None):
        v1 = self.conv(input)
        v2 = v1 + 5
        if padding == None:
            padding = torch.randn(v1.shape)
        v3 = v2 + padding
        return v3
# Inputs to the model
input = torch.randn(1, 10, 64, 64)
