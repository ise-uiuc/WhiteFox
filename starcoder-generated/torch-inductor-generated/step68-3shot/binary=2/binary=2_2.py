
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - torch.randn([8, 8, 1, 1])
        return v2
# Inputs to the model
x = torch.randn(1, 3, 267, 123)
