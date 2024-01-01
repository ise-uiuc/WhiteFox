
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 4)
        self.linear2 = torch.nn.Linear(4, 4)
    def forward(self, x):
        g1 = torch.nn.functional.softmax(self.linear(x))
        g2 = self.linear(g1)
        g3 = self.linear1(g2)
        g4 = self.linear2(g3)
        g4 = torch.nn.functional.elu(g4)
        return g4
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
