
class Model(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.l = torch.nn.Linear(output_size, output_size)
 
    def forward(self, x1, other):
        v1 = self.l(x1)
        v2 = v1 - other
        v3 = torch.relu(v2)
        return v3

# Initializing the model to generate
m = Model(32)

# Inputs to the model
x1 = torch.randn(1, 32)
other = torch.randn(1, 32, requires_grad=True)
