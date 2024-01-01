
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = torch.nn.Linear(3, 3, bias=False)
    def forward(self, x1):
        x2 = self.layer(x1)
        x3 = torch.nn.functional.relu(x2)
        x4 = x1 + x3
        return x4
# Inputs to the model
x1 = torch.randn(1, 3)
