
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
 
    def forward(self, x1, **kwargs):
        v1 = self.linear(x1)
        v2 = F.relu(v1) + 0.5
        v3 = v2 * 0.5
        v4 = torch.relu(v3) + 0.5
        v5 = v4 + 0.5
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
kwargs = {}
kwargs['max_value'] = 1.2
