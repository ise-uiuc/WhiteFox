
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(3, 8)
 
    def forward(self, x1, min_value=-10.4, max_value=0.3):
        v1 = self.dense(x1)
        v2 = torch.clamp(v1, min=min_value)
        v3 = torch.clamp(v2, max=max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
