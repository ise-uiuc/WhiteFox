
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(100, 100, bias=False)
        self.linear2 = torch.nn.Linear(100, 100, bias=True)
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = self.linear2(x)
        return v1 + v2
# Inputs to the model
x = torch.randn(100, 100)
