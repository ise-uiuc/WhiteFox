
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def dot_prod_attention(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output
 
    def mlp(self, x):
        for i in range(2):
            x = torch.relu(x)
            x = torch.linear(x, 8)
        return x
 
    def forward(self, x1):
        x2 = torch.matmul(x1, x1)
        x3 = self.dot_prod_attention(x1, x2, x1, torch.empty([x1.size(0), 1, 1, 1], dtype=torch.float32))
        x4 = x1 + x3
        x5 = self.mlp(x4)
        x6 = torch.sigmoid(x5)
        return x6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
