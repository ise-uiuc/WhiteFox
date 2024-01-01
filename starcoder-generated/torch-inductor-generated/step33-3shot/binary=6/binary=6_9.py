
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(224, 224)
 
        # Initialize the biases in the linear transformation to 0
        self.linear.bias.data.zero_()
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 114
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 224, 224)
