
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, min_value, max_value):
        x2 = self.linear(x1)
        x3 = torch.clamp_min(x2, min_value)
        x4 = torch.clamp_max(x3, max_value)
        return x4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
min_value = 0.924012586117
max_value = 0.304216420217
