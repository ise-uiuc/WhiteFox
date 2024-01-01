
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=16, stride=64, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=16, stride=16, padding=0)
    def forward(self, x1):
        x = F.relu(x1)
        x = x.to(torch.float32)
        x = x.to(device="cuda:0")
        x = x.to(dtype=torch.float64)
        x = torch.add(x, 0.)
        v1 = self.conv1(x)
        v1 = torch.sigmoid(v1)
        v2 = F.relu(v1)
        v2 = v2.to(torch.float32)
        v2 = v2.to(device="cuda:1")
        v2 = v2.to(dtype=torch.float64)
        v2 = torch.add(v2, 0.)
        v3 = self.conv2(v2)
        v3 = torch.sigmoid(v3)
        return v3
# Inputs to the model
x1 = torch.zeros(1, 1, 728, 1248)
