
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, __input_tensor__):
        v0 = torch.cat([x, __input_tensor__], dim=1)
        v1 = torch.cat([v0], dim=1)
        return v1

# Initializing a model 
m = Model()

# Inputs and outputs to the model
x = torch.randn(20, 16, 28, 28)
