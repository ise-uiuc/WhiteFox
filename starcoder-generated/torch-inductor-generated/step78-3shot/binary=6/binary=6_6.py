
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.other = torch.zeros(1, 3)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        return v2

# Initializing the model
m = Model()

# Initializing another tensor other which requires grad
other = torch.Tensor([1, 1, 1])
other.requires_grad_()

# Inputs to the model
x1 = torch.randn(1, 3, 1, 1)
