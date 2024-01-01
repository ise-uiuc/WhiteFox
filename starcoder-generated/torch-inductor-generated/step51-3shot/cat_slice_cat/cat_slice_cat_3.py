
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool=torch.nn.MaxPool2d(3)
        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(1)
 
    def forward(self, x1):
        v1 = self.max_pool(x1)
        v2 = self.adaptive_avg_pool(v1)
        v3 = v2.reshape(0, -1)
        v4 = torch.cat([v2.reshape(0, -1),v2,v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
