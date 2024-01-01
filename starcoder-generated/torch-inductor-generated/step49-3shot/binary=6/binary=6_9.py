
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 6, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.zeros((x1.shape[0], 6, x1.shape[2], x1.shape[3]), dtype=torch.float32, device=x1.device)
        v3 = v1 - v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3, 24, 32)
