
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(12, 1, [1,3,3], stride=1, padding=[2,1,1])
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(1, v1.shape[2], v1.shape[3], v1.shape[4])
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 12, 64, 32, 32)
