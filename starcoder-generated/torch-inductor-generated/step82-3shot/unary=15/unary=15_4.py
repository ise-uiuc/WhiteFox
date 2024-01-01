
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,1,1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        a = 2 * x - 0.5
        v1 = self.conv1(a.view(-1, 1, 28, 28))
        v2 = self.sigmoid(v1).view(-1, 1, 28, 28)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
