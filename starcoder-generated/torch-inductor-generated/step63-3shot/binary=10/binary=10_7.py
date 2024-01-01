
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3*2*4, 1)
        self.other = torch.randn(1, 3*2*4)
        # Please make sure that the tensors self.linear.weight and self.other are identical in shape
 
    def forward(self, x1):
        v1 = self.linear(x1.flatten(start_dim=1))
        v2 = v1 + self.other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 2, 4)
