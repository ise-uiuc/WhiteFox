
class Model(torch.nn.Module):
    def __init__(self, input_size=1):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, 1, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model(8)

# Inputs to the model
x1 = torch.randn(20, 8)
