
class Model(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.l1 = torch.nn.Linear(1, 1)
        self.l2 = torch.nn.Linear(1, 1)
        self.dropout = dropout
    def forward(self, x):
        y = self.l1(x)
        y = F.functional[self.dropout](y)
        return self.l2(y)
# Inputs to the model
x = torch.ones((1, 1), requires_grad=True)
