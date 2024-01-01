
class Model(torch.nn.Module):
    def __init__(self, input_num_dims, output_num_dims):
        super().__init__()
        self.linear = torch.nn.Linear(input_num_dims, output_num_dims)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model(32, 16)

# Inputs to the model
x1 = torch.randn(2, 32)
