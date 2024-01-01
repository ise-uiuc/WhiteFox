
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(11, stride=1)
 
    def forward(self, x1):
        v1 = self.avgpool(x1)
        v2, _ = torch.max(v1, dim=2, keepdim=False)
        v3, _ = torch.max(v2, dim=3, keepdim=False)
        return v3
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
