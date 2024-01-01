
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2):
        v1 = torch.addmm(x1, x2, self.conv.weight.data)
        v2 = torch.cat([v1], dim=1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1,256,14,14)
x2 = torch.randn(256,512,1,1)
