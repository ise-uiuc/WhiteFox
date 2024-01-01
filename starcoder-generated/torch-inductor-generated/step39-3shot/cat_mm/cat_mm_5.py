
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 5, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(5, 7, 1, bias=False)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        w = list(v2.size())
        w[1] = 1
        return v2.view(tuple(w))
# Inputs to the model
x = torch.randn(2,2,2,2)
