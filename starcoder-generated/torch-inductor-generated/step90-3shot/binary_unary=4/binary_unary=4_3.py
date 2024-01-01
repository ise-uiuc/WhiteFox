
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, x1, x_other):
        t1 = self.linear(x1)
        t2 = t1 + x_other
        t3 = torch.relu(t2)
        return t3

# Initializing the model
m = Model()

# Inputs to the model, "x_other" is a tensor, "x1" as well
x_other = torch.randn(1, 5)
x1 = torch.randn(1, 10)
