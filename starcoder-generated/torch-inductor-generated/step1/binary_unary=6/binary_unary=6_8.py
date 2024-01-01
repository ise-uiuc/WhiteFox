 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)

# Other input to the model, the value of which will vary during model's execution
other = torch.randn(1, 20)
