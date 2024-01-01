
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x):
        v = self.linear(x)
        v2 = torch.clip_min(v, self.min_value)
        v3 = torch.clip_max(v2, self.max_value)
        return v3
    
# Initializing the model with specified minimum and maximum values
min_value = -1.0
max_value = 1.0
m = Model(min_value, max_value)

# Inputs to the model
x = torch.randn(1, 8)
