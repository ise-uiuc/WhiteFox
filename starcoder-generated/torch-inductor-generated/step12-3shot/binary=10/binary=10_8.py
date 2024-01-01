
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 120
        output_size = 84
        weights = torch.empty(output_size, input_size)
        self.linear = torch.nn.Linear(input_size, output_size, bias=False)
        self.linear.weight.data = weights
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        return v2


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 120)
x2 = torch.randn(1, 84)
