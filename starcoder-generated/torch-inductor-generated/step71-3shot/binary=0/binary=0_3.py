
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(87, 2, 1, stride=2, padding=1)
    def forward(self, input):
        v1 = self.conv(input)
        w1 = torch.sum(v1, dim=[0,2,3])
        v2 = v1.squeeze(dim=0)
        z1 = torch.cat([w1,v2],-1)
        return z1
# Inputs to the model
input = torch.randn(1, 87, 28, 28)
