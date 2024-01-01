
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 8)
 
    def forward(self, x1):
        v5 = torch.add(x1, x1) # Add the input tensor twice, so the tensor has the same shape as the output tensor of the linear transformation
        v1 = self.linear(v5)
        v2 = torch.sigmoid(v1)
        v3 = torch.mm(v5, v5.t())
        v4 = v3 * v2
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 12)
