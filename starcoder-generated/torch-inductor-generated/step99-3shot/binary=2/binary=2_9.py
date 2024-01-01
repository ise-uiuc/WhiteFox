
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 512, (1, 3), stride=(1, 1), padding=(0, 1))
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - x
        return v2
# Outputs of this model:
v = torch.abs(torch.mean(v1 - x, (2,3)))
