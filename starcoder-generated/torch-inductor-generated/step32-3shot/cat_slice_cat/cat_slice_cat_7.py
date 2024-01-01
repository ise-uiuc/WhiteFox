
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
 
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.conv2(t1)
        t3, _ = torch.max(t2, dim=-1)
        t4 = torch.cat([t1, t3], dim=1)
        return t4
 
# Initializing the model
m2 = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
