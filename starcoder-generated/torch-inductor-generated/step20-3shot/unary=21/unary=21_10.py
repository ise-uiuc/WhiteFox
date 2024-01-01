
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv = nn.Conv2d(3, 32, (2, 2))
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 200, 200)
