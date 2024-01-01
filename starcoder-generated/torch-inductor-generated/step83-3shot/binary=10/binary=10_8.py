
class Model(torch.nn.Module):
    def __init__(self, __random_name__):
        super().__init__()
        self.linear = torch.nn.Linear(128, 256)
 
    def forward(self, x1, __random_name__):
        v1 = self.linear(x1)
        v2 = v1 + __random_name__
        return v2

# Initializing the model
m = Model(__random_name__)

# Inputs to the model
x1 = torch.randn(1, 128)
