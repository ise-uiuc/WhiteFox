
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
    
    def forward(self, input_tensor):
        t1 = self.linear(input_tensor)
        return t3.relu()

# Initializing the model
m = Model(torch.ones(3))

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
