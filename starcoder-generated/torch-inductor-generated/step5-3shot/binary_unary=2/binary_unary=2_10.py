
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, stride=1, padding=4) #kernel_size=3, stride=1, padding=4
    def forward(self, x1):
        v1 = F.relu(self.conv(x1))
        v2 = v1 - 0.5 # 0.5 is the scalar
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
