
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 10) # The linear layer acts as a matrix multiplication between its input tensor and this matrix
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.cat([v1], dim=1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
