
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(1, 5)
 
    def forward(self, input):
        y1 = self.linear(input)
        y2 = torch.clamp_min(y1,min_value)
        y3 = torch.clamp_max(y2,max_value)
        return y3

# Initializing the model
min_value = 0.1
max_value = 0.9

m = Model(min_value, max_value)

# Inputs to the model
x1 = torch.randn(2, 1)
