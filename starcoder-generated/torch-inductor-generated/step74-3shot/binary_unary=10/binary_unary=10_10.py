
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 8)
        self.linear2 = torch.nn.Linear(8, 64)
 
    def forward(self, x1):
        t1 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        t2 = t1 + self.linear2.weight
        t3 = torch.nn.functional.relu(t2)
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
