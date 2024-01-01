
class Model(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_p):
        super().__init__()
        self.qkv_proj = torch.nn.Linear(hidden_size, num_heads * 3)
        self.dropout_p = dropout_p
 
    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        qkv = self.qkv_proj(x1)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        scale_factor = hidden_size**(-0.5)
        inv_scale_factor = 1 / scale_factor
        q = q.matmul(k.transpose(-2, -1))
        q = q.div(inv_scale_factor)
        softmax_q = q.softmax(dim=-1)
        dropout_q = torch.nn.functional.dropout(softmax_q, p=self.dropout_p)
        output = dropout_q.matmul(v)
        return output

# Initializing the model
m = Model(hidden_size=16, num_heads=4, dropout_p=0.1)

# Inputs to the model
x1 = torch.randn(4, 16, 32)
