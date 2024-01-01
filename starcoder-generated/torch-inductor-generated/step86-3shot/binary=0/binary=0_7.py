
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, stride=1, padding=1)
    def forward(self, x, pad_val=None):
        v1 = self.conv(x)
        if pad_val == None:
            pad_val = torch.randn(v1.shape)
        v2 = v1 + pad_val
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
