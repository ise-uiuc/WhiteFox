
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 1024)
 
    def forward(
        self,
        x,
        other,
    ):
        v1 = self.linear(x)
        v3 = v1 + other
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 1024)
other = torch.randn(1, 1024)
