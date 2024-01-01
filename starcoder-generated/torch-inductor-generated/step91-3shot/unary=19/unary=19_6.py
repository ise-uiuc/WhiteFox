
class Model(torch.nn.Linear):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model(3, 8)

# Inputs to the model
x1 = torch.randn(1, 3)
