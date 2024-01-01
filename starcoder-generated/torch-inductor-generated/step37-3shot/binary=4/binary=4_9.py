
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1)
        self.layer0 = torch.nn.Linear(8, 8, bias=False)
    
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1.view(v1.shape[0], -1)
        v3 = self.layer0(v2)
        return torch.cat((v1, v3), 1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
