
class Model(torch.nn.Module):
    def __init__(self, __input_size1, __input_size2, __output_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(__input_size1, __output_size)
        self.linear2 = torch.nn.Linear(__input_size2, __output_size)
 
    def forward(self, x1, x2):
        v1 = self.linear1(x1)
        v2 = v1 + x2
        v3 = v2.relu()
        return v3

# Initializing the model
m = Model(32, 16, 8)

# Inputs to the model
x1 = torch.randn(128, 32)
x2 = torch.randn(128, 16)
