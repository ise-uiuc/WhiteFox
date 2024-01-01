
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1,5)
        self.linear_copy = copy.deepcopy(self.linear)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.add(v1, self.linear_copy)
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model with weights randomized
m = Model()

# We can access the randomized weights with
print(m.linear.weight.data)

# We can set the weights we want
weights = torch.tensor([[1.9]])
m.linear.weight = torch.nn.parameter.Parameter(weights)

# Inputs to the model
x1 = torch.tensor([[3.2]])
