
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.randn_like(v1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model and the associated groundtruth tensors
x1 = torch.randn(1, 3, 64, 64)
GT1 = torch.randn(1, 16, 32, 32)
__output__, = m(x1)

