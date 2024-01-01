
class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.other = torch.randn(output_size, )
 
    def forward(self, x2):
        v0 = self.linear(x2)
        v1 = v0 + self.other
        return v1

# Initializing the model
m = Model(10, 10)

# Inputs to the model
x2 = torch.randn(1, 10)
