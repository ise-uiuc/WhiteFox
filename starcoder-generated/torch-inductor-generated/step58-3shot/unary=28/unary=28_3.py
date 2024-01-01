
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
 
    def forward(self, x1, min_value, max_value):
        t1 = self.linear(x1)
        t2 = torch.clamp_min(t1, min_value)
        t3 = torch.clamp_max(t2, max_value)
        return t3

# Initializing the model
m = Model()

# Inputs to the model (random tensors)
x1 = torch.randn(2, 5)
min_value = 0.01
max_value = 0.99
