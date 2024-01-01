
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(369, 453)
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 18.856593705474854
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.ones(1, 1, 1, 1)
