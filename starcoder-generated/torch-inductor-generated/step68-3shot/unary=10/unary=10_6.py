
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_in = torch.nn.Linear(10, 10)
        self.linear_out = torch.nn.Linear(10, 5)
        self.l4 = torch.nn.ReLU6()
 
    def forward(self, x1):
        l1 = self.linear_in(x1)
        l2 = l1 + 3
        l3 = torch.clamp_min(l2, 0)
        l3 = self.l4(l3)
        l4 = l3 / 6
        l4 = self.linear_out(l4)
        return l4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 10)
x2 = torch.randn(5, 10)
