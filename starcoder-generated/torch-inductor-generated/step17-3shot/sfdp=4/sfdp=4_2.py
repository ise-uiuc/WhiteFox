
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
 
        self.q = torch.nn.Linear(64, 8)
        self.k = torch.nn.Linear(64, 8)
        self.v = torch.nn.Linear(64, 8)
 
    def forward(self, x3):
        v3 = self.q(x3)
        v4 = self.k(x3)
        v6 = self.v(x3)
 
        v0 = batch_dot(v3, v4, axes=(2, 2)) / math.sqrt(v3.shape[-1])
        v1 = v0 + self.v.bias.unsqueeze(0).unsqueeze(1)
        v2 = self.softmax(v1)
        v5 = batch_dot(v2, v6, axes=(1, 2))
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(1, 16, 64)
