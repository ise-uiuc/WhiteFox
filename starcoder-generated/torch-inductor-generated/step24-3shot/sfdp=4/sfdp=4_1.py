
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 4)
 
 
    def forward(self, q, k, v):
        q, k, v = self.proj(q), self.proj(k), self.proj(v)
        q *= 0.5
        k *= 0.5
 
        dot_product = torch.matmul(q, k.transpose(-2, -1))
        attn_mask = dot_product.new_ones(dot_product.shape)
        for i in range(23):
            attn_mask[:, :, i, i] = float('-inf')
 
        return torch.matmul(torch.softmax(dot_product + attn_mask, dim=-1), v)

# Initializing the model
m = Model()

# Inputs of the model
q = torch.randn(1, 1, 3)
k = torch.randn(1, 2, 3)
v = torch.randn(1, 2, 3)
__o = m(q, k, v)

