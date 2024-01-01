
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 8)
  
    def forward(self, x1):
        d1 = self.linear(x1)
        d2 = d1 + torch.tensor([[1., 2., 3., 0.], [4., 5., 6., 0.]])
        return d2.clamp(min=0.)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
