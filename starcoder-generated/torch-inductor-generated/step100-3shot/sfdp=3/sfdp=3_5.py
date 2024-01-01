
class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.output_size = output_size
 
        self.weight = torch.nn.Parameter(torch.randn((input_size, output_size)))
 
    def forward(self, x):
        y = torch.matmul(x, self.weight)
        z = torch.nn.functional.softmax(y, dim=-1)
        return z

# Initializing the model
input_size = 3
output_size = 5
m = Model(input_size, output_size)

# Input to the model
x = torch.randn(4, 3)
