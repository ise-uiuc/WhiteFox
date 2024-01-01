
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor, other):
        t1 = self.linear(input_tensor)
        return t1 + other

# Initializing the model
m = MyModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)
other = torch.zeros(1, 3, 64, 64)
