
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 5, stride=2, padding=2)
    def forward(self, x1, x2=0, padding=None):
        v1 = self.conv(x1)
        if padding == None:
            padding = torch.randn(v1.shape)
        v2 = v1 + x2
        v3 = v2 + padding
        return v3
# Inputs to the model
x1 = torch.randn(1, 5, 300, 300)
