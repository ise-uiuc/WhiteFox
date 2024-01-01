
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(32, 8)
        self.key = torch.nn.Linear(32, 8)
        self.value = torch.nn.Linear(32, 8)
 
    def forward(self, x1):
        qk = self.query(x1) @ self.key.T() / math.sqrt(self.query.out_features)
        v1 = qk + 0.5
        v2 = torch.softmax(v1, dim=-1)
        output = v2 @ self.value(x1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
