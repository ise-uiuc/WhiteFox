
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 128, 1, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.ones(v1.shape)
        v2 = v1 + other
        v3 = self.pool(v2)
        return v3
# Inputs to the model
x1 = torch.ones(45, 1, 8192)
