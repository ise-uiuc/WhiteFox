
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(14*8*8, 64*2)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v3 = torch.relu(v1 + torch.mean(x2, dim=1, keepdim=True))
        return v3.view(v3.shape[0], 64, 2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 14*8*8)
x2 = torch.randn(1, 64*8*8)
