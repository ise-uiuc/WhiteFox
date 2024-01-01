
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Other tensor (for demonstration purposes, it simply copies the value of the input tensor)
other = torch.randn(8, 128)
# Inputs to the model
x1 = torch.randn(8, 128)
