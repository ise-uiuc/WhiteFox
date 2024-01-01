
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, stride=2, padding=1))
    def forward(self, x1):
        v1 = self.features(x1)
        v2 = torch.mean(v1, dim=(2,3), keepdims=False)
        
        v3 = self.features(x1)
        v4 = torch.mean(v3, dim=(2,3), keepdims=False)
        v5 = self.features(x1)
        v6 = torch.mean(v5, dim=(2,3), keepdims=False)
        v7 = v4 - v6
        v8 = F.relu(v7)
        v9 = v2 - 0.5
        v10 = F.relu(v9)
        v11 = v10 + v8
        v12 = torch.abs(v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
