
class Model(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer2 = nn.Linear(10, 10)
 
    def forward(self, x1):
        v1 = self.layer2(x1)
        v2 = v1 * torch.clamp(torch.min(torch.max(v1 + 3, 0), 6), min=0, max=6)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 10)
