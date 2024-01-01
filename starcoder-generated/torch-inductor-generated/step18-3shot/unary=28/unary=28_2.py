 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = F.relu6(v1)
        v3 = F.hardswish(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
min_value = 0.25
max_value = 0.8
