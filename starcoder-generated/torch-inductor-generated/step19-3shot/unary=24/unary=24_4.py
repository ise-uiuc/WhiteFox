
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.mask = torch.zeros((2, 8, 1, 1), dtype=torch.bool)
    def forward(self, x1):
        for i in range(2):
            self.mask[i,...] = True
        v1 = self.conv(x1)
        v2 = torch.where(self.mask, v1, v1 * 0.1)
        # print(v2.shape)
        return v2
# Inputs to the model
x1 = torch.randn(2, 3, 256, 256)
