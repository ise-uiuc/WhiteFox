
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 32, 1, stride=1, padding=1),
                                   nn.Sigmoid(),
                                   nn.Sigmoid())

    def forward(self, x):
        return self.model(x)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
