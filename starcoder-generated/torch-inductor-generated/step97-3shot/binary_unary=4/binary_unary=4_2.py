
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 16)
        self.linear2 = torch.nn.Linear(4, 16)

    def forward(self, x, other):
        l1 = self.linear1(x)
        l1_plus_other = l1 + other
        l2 = self.linear2(x)
        l2_plus_other = l2 + other
        return F.relu(l1_plus_other), F.relu(l2_plus_other)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(4)
other = torch.randn(16)
__output1__, __output2__ = m(x, other=other)
