
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.h1 = nn.Linear(33, 26)
        self.h2 = nn.Linear(33, 26)
        self.h3 = nn.Linear(33, 26)
        self.h4 = nn.Linear(33, 26)
    def forward(self, x1, x2, x3, x4, x5, x6):
        h1 = self.h1(x1 + x3 + x5)
        h2 = self.h2(x2 + x4 + x6)
        return h1 + h2
# Inputs to the model
x1 = torch.randn(7, 7)
x2 = torch.randn(7, 7)
x3 = torch.randn(7, 7)
x4 = torch.randn(7, 7)
x5 = torch.randn(7, 7)
x6 = torch.randn(7, 7)
