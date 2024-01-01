
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1[:,:,:,1:4]
        v3 = v2[:,:,:,::4] + v2[:,:,:,2:5] + v2[:,:,:,12:15] + v2[:,:,:,66:69]
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
