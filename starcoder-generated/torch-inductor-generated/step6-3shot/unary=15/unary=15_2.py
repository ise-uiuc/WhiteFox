
class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 16, 4, padding=2)
        self.conv1 = torch.nn.Conv2d(3, 16, 4, padding=2)
    def forward(self, x1):
        v0 = torch.cat([x1,x1],0)
        v1 = torch.cat([x1,x1],1)
        v2 = torch.cat([x1,x1],2)
        v3 = torch.cat([x1,x1],3)
        v4 = torch.cat([v0,v1,v2,v3],1)
        v5 = self.conv1(v4)
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
