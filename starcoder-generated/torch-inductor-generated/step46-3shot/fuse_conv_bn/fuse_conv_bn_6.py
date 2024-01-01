
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 3)
    def forward(self, x):
        x = self.linear1(x)
        return x.max()
# Inputs to the model
x = torch.rand([2, 3])
