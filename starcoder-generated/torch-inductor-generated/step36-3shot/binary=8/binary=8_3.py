
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x):
       v = self.conv1(x)
       w = self.conv1(x)
       v = v.view(v.shape[0], v.shape[1], -1).sum(-1)
       w = w.view(w.shape[0], w.shape[1], -1).sum(-1)
       w = w.view(v.shape[0], -1, 2, 2).sum(-1).sum(-1)
       return w
# Inputs to model
x = torch.randn(1, 3, 64, 64)
