
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 4)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model; the value of "other" should be specified on initialization
m = Model(other=torch.tensor(1))

# Inputs to the model
x1 = torch.randn(1, 12)
