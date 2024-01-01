
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 4)
        self.linear2 = torch.nn.Linear(3, 2)
 
    def forward(self, a1, a2):
        c1 = torch.cat((a1, a1))
        l1 = self.linear1(a1)
        l2 = self.linear2(a2)
        l3 = l1 + l2
        l4 = l3 + c1
        return l4

# Initializing the model
m = Model()

# Inputs to the model
a1 = torch.randn(5, 3)
a2 = torch.randn(5, 3)
