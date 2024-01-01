
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(2, 7)
 
    def forward(self, x):
        w1 = self.linear(x)
        w2 = torch.clamp_min(w1, min_value=self.min_value)
        w3 = torch.clamp_max(w2, max_value=self.max_value)
        return w3

# Initializing the model
m = Model(3.1, 4.1)

# Inputs to the model
x = torch.randn(7, 2, 11, 13)
