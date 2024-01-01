
class Model(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear_1 = nn.Linear(3, 3)
        self.linear_2 = nn.Linear(3, 3)

    def forward(self, x1, x2, inp):
        out = self.linear_1(x1) + self.linear_2(x2) + inp
        return out.view(out.size(0), -1)
# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 3)
inp = torch.randn(1, 3)
