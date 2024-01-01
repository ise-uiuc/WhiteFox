
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 5)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
