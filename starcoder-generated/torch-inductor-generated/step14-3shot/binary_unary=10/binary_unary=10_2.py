
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + torch.arange(8).view(8).to(v1)
        return nn.ReLU()(v2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
