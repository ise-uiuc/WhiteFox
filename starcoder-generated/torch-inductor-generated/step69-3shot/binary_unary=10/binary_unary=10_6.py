
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(200,20)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + x1
        v3 = torch.relu(v2)
        return v3


# Initializing the model
m = Model()

# Initializing tensor for inputs to the model
x1 = torch.randn(1, 200)

# Inputs to the model
