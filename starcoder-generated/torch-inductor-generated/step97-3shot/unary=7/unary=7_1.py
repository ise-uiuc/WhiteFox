
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(7, 1, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * torch.clamp(v1.sum(dim=1, keepdim=True), 0, 6)
        v3 = v1 / 6
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 7)
