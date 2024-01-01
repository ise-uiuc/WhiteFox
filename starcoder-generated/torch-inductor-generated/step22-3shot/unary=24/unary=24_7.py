
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, 7, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * torch.tensor([3.57680109, 0.20652153, 0., 0., 0.13794327, 0., 0.08564419, 0.01622201]) # Some random value
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 8, 20, 20)
