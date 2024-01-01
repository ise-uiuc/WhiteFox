
class Model(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.q = torch.nn.Linear(hidden_size, 32)
        self.k = torch.nn.Linear(hidden_size, 32)
 
    def forward(self, x1, x2):
        q1 = self.q(x1)
        k1 = self.k(x2)
        v1 = torch.matmul(q1, k1.transpose(0, 1))
        v1 = v1 / math.sqrt(q1.size(2))
        v1 = v1 + attn_mask
        v1 = torch.softmax(v1, dim=-1)
        v1 = torch.dropout(v1, 0.0)
        v1 = torch.matmul(v1, x1)
        return v1

# Initializing the model
m = Model(1024)

# Inputs to the model
x1 = torch.randn(25, 1024)
x2 = torch.randn(25, 1024)
