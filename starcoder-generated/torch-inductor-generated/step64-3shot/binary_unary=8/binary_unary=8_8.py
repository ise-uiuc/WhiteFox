
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(8,1,2,(3,3,3),(1,1,1),1,2,2,True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = torch.relu(v1 + v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 8, 10, 10, 10)
