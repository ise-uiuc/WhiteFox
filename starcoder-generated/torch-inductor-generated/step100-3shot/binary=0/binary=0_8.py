
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 16, 3, padding=1)
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = torch.relu(v1 + other)
        return v2 if v2.size(3) == 8 and v2.size(2) == 8 else v2.sum(dim=(2, 3, 4)) 
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64, 64)
