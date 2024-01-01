
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 4)
 
    def forward(self, x1):
        v1 = x1.mean(dim=1, keepdim=True)
        v2 = self.linear1(v1)
        v3 = v2 * torch.clamp(v2, 0, 6) + 3
        v4 = v3 / 6
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 2)
