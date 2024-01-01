
class Model(torch.nn.Module):
    def __init__(self, value=0.12):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 2, stride=1, padding=1)
        #self.conv = torch.nn.Conv2d(10, 20, (1, 2))
        self.value = value
    def forward(self, x1):
        #v1 = self.conv(x1)
        #v2 = torch.clamp_min(v1, self.value)
        v2=self.conv(x1)
        v3 = torch.clamp_min(v2, self.value)
        return v3
# Inputs to the model
value = 0.12
x1 = torch.randn(1, 3, 64, 64)
