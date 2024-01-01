
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.negative_slope = 0.1
 
    def forward(self, x1):
        y = torch.nn.Linear(3, 8  )
        return y.forward(x1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
