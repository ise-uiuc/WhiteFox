
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        return v3


# Initializing the model
m = Model1()

# Inputs to the model
x2 = torch.randn(1, 10)
