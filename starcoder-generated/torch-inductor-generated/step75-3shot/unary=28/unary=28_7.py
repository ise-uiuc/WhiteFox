
class Model(torch.nn.Module):
    def __init__(self):
       super().__init__()
       self.linear = torch.nn.Linear(3, 5, bias=False)
       self.clamp_min_fn = torch.nn.functional.relu
       self.clamp_max_fn = torch.nn.functional.sigmoid

    def forward(self, x1, min_value, max_value):
        v1 = self.linear(x1)
        v2 = self.clamp_min_fn(v1, min_value)
        v3 = self.clamp_max_fn(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
min_value = torch.tensor(0, dtype=torch.float)
max_value = torch.tensor(1, dtype=torch.float)
