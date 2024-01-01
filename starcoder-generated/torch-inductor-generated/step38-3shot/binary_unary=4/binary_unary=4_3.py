
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 7)
 
    def forward(self, x, other = torch.empty(1)):
        v1 = self.linear(x)
        if other.shape[1]!= 1:
            raise ValueError
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
