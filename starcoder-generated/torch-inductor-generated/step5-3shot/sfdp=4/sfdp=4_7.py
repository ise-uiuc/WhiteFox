
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(20, 32)
        self.linear = torch.nn.Linear(32, 32)
 
    def forward(self, x1, x2, x3, x4):
        v1 = self.embedding(x1)
        v2 = torch.transpose(self.embedding(x2), -2, -1)
        v3 = v1 @ v2
        v4 = torch.unsqueeze(v3 / math.sqrt(v3.size(-1)), dim=-1)
        v5 = v4 + x3
        attn_weights = torch.softmax(v5, dim=-1)
        v6 = attn_weights @ x4
        v7 = self.linear(torch.flatten(v6, start_dim=1))
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randint(0, 20, (1, 20))
x2 = torch.randint(0, 20, (1, 32, 2))
x3 = torch.zeros(1, 20, 2)
x4 = torch.randn(1, 20, 32)
