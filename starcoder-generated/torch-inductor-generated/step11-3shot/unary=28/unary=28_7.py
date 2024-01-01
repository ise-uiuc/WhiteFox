
class Model(torch.nn.Module):
    def __init__(self, min_value):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(2, 3, bias=False)
        self.min_value = min_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
min_value = 0.1
m = Model(min_value)

# Inputs to the model
x1 = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32)
