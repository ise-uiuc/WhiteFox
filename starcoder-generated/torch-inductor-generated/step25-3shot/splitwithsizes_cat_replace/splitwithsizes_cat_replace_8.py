
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(1, 168, 45, 35, 46), torch.nn.Conv2d(168, 107, 3, 21, 8))
        self.pad = torch.nn.Sequential(torch.nn.ConstantPad3d((0, 160, 100, 50), value=1.69117943))
        self.relu = torch.nn.ReLU(inplace=True)
        self.reshape = torch.nn.Sequential(torch.nn.Bilinear(490, 62, 482, bias=False))
        self.gelu = torch.nn.GELU()
    def forward(self, x1):
        x2 = self.features(x1)
        x2 = self.pad(x2)
        x2 = torch.clamp(x2, 0)
        x2 = self.relu(x2)
        x2 = torch.split(x2, [1, 1, 1, 1], dim=1)
        x2 = self.reshape(x2)
        x2 = torch.t(x2)
        x2 = self.gelu(x2)
        return (x2, x2)
# Inputs to the model
x1 = torch.randn(1, 1, 789, 215)
