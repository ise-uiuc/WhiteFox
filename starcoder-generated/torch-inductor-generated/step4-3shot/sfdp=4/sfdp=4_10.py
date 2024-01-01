
class Model(torch.nn.Module):
    def __init__(self, n_key, n_value, n_hidden):
        super().__init__()
        self.linear1 = torch.nn.Linear(n_value, n_hidden)
 
    def forward(self, v1, v2):
        v3 = v2.matmul(v1.transpose(-2, -1))
        v4 = v3 / math.sqrt(v1.size(-1))
        v4 = v4 + attn_mask
        v5 = torch.softmax(v4, dim=-1)
        v6 = v5.matmul(v2)
        v7 = self.linear1(v6)
        return v7

# Initializing the model
m = Model(1, 2, 3)

# Inputs to the model
v1 = torch.randn(1, 2, 2)
v2 = torch.randn(1, 2, 3)
