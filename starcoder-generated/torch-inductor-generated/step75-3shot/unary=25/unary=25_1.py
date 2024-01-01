
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        
    def forward(self, x1):
        v1 = F.dropout(x1)
        v2 = self.linear(v1)
        v3 = v2 < 0
        v4 = v2 * negative_slope
        v5 = torch.where(v3, v2, v4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, requires_grad=True)
