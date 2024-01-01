
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.c1 = nn.ConvTranspose2d(19, 48, (8, 6), 1, (3, 2), 1)
    def forward(self, x14):
        r1 = self.c1(x14)
        r2 = r1 > 0
        r3 = r1 * 0.105
        r4 = torch.where(r2, r1, r3)
        return torch.nn.functional.adaptive_avg_pool2d(torch.nn.functional.relu(r4), 5)
# Inputs to the model
x14 = torch.randn(487, 19, 17, 9)
