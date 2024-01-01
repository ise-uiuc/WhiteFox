
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.clamp_min(v1, min_value=0.1)
        v3 = torch.clamp_max(v2, max_value=0.3)
        return v3

# Initializing the model
m = Model()
m.linear.bias.data = torch.tensor([0.5, -0.1, 0.3, 0.2], dtype=torch.float)
m.linear.weight.data = torch.tensor([[0.3, 0.3], [0.2, 0.1], [0.8, 0.3], [0.4, 0.6], [0.1, 0.2], [-0.5, 0.8], [0.7, 0.6], [0.8, 0.9], [-0.5, 0.3], [0.2, 0.4], [0.1, 0.7], [-0.5, 0.2], [0.0, 0.3], [-0.7, 0.5], [-0.8, 0.6], [-0.4, 0.2]], dtype=torch.float)

# Inputs to the model
x = torch.empty(32, 16)
