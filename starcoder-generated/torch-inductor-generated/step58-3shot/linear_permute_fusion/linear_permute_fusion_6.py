
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x1):
        x2 = torch.zeros([2,2])
        v2 = x2.view([1, 2, 2])
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
