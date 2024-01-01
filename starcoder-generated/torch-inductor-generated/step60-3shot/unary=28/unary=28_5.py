
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(224, 1000)
 
    def forward(self, x1, min_value, max_value):
        v1 = self.linear(x1)
        v2 = torch.clamp(v1, min_value, max_value)
        return v1

# Initializing the model
m = MyModel(0.2, 0.9)

# Inputs to the model
x1 = torch.randn(200, 243, 224, 224)
