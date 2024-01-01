
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3,4)
 
    def forward(self, input, min_value, max_value):
        v1 = self.linear(input)
        v2 = torch.clamp_min(v1, min_value)
        output = torch.clamp_max(v2, max_value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
__input__ = torch.randn(1, 3)
__min_value__ = 0.5
__max_value__ = 1.0
