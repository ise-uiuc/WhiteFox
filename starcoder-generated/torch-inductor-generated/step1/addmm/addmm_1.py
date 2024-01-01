
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 8)
        self.linear2 = torch.nn.Linear(8, 8)
 
    def forward(self, x, inp):
        v1 = self.linear1(x)
        v2 = self.linear2(v1)
        v3 = torch.mm(v2, v2)
        return inp + v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
