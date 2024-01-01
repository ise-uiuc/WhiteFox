
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(10000, 10)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
self.other = 1e-10
m = Model(self.other)

# Inputs to the model
x1 = torch.randn(1, 10000)
self.x1 = x1

# Calling the function to get the value for 't1'
v1 = self.linear(self.x1)
print(v1)
print(self.x1.grad.data)

# Calling the function to get the value for 't2'
v2 = v1 - self.other
print(v2)
print(self.x1.grad.data)

# Calling the function to get the value for 't3'
v3 = torch.nn.functional.relu(v2)
print(v3)

