
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
    def forward(self, x):
        v = self.conv(x)
        v = v + -138
        return v
# Inputs to the model
x = torch.randn([1,3,16,16])
