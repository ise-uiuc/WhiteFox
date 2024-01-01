
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.min_value = 1.5
        self.max_value = 4.8
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clip_min(v1, self.min_value)
        v3 = torch.clip_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
