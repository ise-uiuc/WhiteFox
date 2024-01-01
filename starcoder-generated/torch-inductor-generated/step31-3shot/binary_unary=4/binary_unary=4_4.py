
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 1)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
# The second input tensor can be any random tensor with the same number of rows with the output of linear transformation of the first tensor.
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 1)
