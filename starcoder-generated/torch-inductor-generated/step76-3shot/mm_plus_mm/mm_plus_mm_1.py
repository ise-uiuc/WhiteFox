
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = torch.nn.Linear(10, 10, bias=False)
        self.layer2 = torch.nn.Linear(10, 10, bias=True)
        self.layer3 = torch.nn.Linear(10, 10, bias=False)

    def forward(self, x1, x2):
        h1 = self.layer1(x1)
        h2 = self.layer2(x2)
        h3 = self.layer3(h1 + h2)
        return h1 + h3 + h3
# Inputs to the model
x1 = torch.randn(5, 10)
x2 = torch.randn(5, 10)
