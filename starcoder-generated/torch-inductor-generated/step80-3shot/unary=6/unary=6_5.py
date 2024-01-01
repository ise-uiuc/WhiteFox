
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layerA = nn.Sequential(
            nn.Conv2d(3, 1, 13, stride=2, padding=4),
            nn.ReLU(),
            nn.ConvTranspose2d(1, 4, 5, stride=2, padding=3),
            nn.Softmax(dim=1)
        )
    def forward(self, x1):
        t1 = self.layerA(x1)
        t2 = t1 + 3
        t3 = torch.clamp(t2, 0, 6)
        t4 = t1 * t3
        t5 = t4 / 6
        return t5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
