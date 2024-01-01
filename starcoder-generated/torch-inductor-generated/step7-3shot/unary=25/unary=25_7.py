
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 6)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = (v1 > 0).float()
        v3 = v1 * 0.01
        v4 = torch.where(v2.unsqueeze(2).unsqueeze(3).expand(v2.shape[0], v2.shape[1], v2.shape[2], 6), v1, v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
