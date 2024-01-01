
class Model(torch.nn.Module):
    def __init__(self, n_heads=8, dim=64):
        super().__init__()
        self.dot_product_attention = lambda x: torch.matmul(x, x.transpose(-2, -1))

        self.multi_head_attention = torch.nn.MultiheadAttention(dim, n_heads)
 
    def forward(self, q, k, v):
        qk = self.dot_product_attention(q, k)
        scaled_qk = qk * 1/math.sqrt(self.dim)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = self.multi_head_attention(v=dropout_qk, q=dropout_qk)
        return output

# Initializing and inputing to the model
m = Model()

# Inputs to the model
q = torch.randn(2, 8, 64)
k = torch.randn(2, 6, 64)
v = torch.randn(2, 6, 64)
