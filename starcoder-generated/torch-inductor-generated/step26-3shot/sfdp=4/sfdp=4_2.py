
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_in, d_model, num_heads):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.Q = torch.nn.Linear(d_in, d_model, bias=False)
        self.K = torch.nn.Linear(d_in, d_model, bias=False)
        self.V = torch.nn.Linear(d_in, d_in, bias=False)
 
    def forward(self, x1):
        _, n, _ = x1.shape
        q = self.Q(x1)
        k = self.K(x1).permute(0, 2, 1)
        v = self.V(x1).permute(0, 2, 1)
        qk = q @ k / math.sqrt(q.size(-1))
        attn_weight = torch.softmax(qk, dim=-1)
        output = torch.matmul(attn_weight, v)
        return output

# Initializing the model
m = MultiHeadAttention(3, 6, 2)

# Inputs to the model
x1 = torch.randn((1, 1, 3))
