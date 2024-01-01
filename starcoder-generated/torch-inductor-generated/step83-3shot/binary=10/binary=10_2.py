
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x):
        v1 = self.linear(input_tensor)
        return v1 + __other__

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
y = torch.randn(1, 10)
