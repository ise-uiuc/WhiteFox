
class Model(torch.nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.linear = torch.nn.Linear(num_in, num_out)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model(28*28, 10)

# Inputs to the model
x1 = torch.randn(5, 28, 28)
