
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, x, min_value=None, max_value=None):
        v1 = self.linear(x)
        if min_value!= None and max_value!= None:
            v2 = torch.clamp(v1, min=min_value, max=max_value)
        elif min_value!= None:
            v2 = torch.clamp(v1, min=min_value)
        elif max_value!= None:
            v2 = torch.clamp(v1, max=max_value)
        else:
            v2 = v1
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
min_value = 1
max_value = 4
