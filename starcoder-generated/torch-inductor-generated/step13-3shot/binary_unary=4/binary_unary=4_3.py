
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x):
        ret = self.linear(x)
        t2 = ret + other
        return torch.nn.functional.relu(t2)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(3, 8)
other = torch.randn(1, 8).abs().sum(0)
