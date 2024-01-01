
class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return t3

# Initializing the model
input_size = 8
output_size = 16
m = Model(input_size, output_size)

# Inputs to the model
x1 = torch.randn(1, input_size)
