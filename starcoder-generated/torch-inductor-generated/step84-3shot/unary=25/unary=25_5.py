
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.randn(8, 8, 3, 3, 3)
        self.b = torch.randn(8, 8)
 
    def forward(self, x1):
        v1 = F.conv3d(x1, self.w, self.b, stride=1, padding=1)
        v2 = v1 > 0
        v3 = v1 * 0.1
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 32, 32, 32)
