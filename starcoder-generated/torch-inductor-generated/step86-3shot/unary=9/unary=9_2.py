
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 13, stride=6, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 6, padding=3)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, padding=1)
    def forward(self, x1):
        h1 = self.conv1(x1)
        h2 = h1 + 3
        h3 = torch.clamp(h2, min=0, max=6)
        h4 = torch.div(h3, 6)
        h5 = self.conv2(h4)
        h6 = h5 + 3
        h7 = torch.clamp(h6, min=0, max=6)
        h8 = torch.div(h7, 6)
        h9 = self.conv3(h8)
        h10 = h9 + 3
        h11 = torch.clamp(h10, min=0, max=6)
        h12 = torch.div(h11, 6)
        return h12
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
