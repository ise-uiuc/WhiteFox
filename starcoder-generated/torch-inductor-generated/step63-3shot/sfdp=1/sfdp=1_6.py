
class Model(torch.nn.Module):
    def __init__(self, d_model, n_head, d_head, dropout):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.n_head = n_head
        self.d_head = d_head
        self.scale_factor = (d_head ** -0.5)
        self.q = torch.nn.MultiheadAttention()
        self.k = torch.nn.MultiheadAttention()
        self.v = torch.nn.MultiheadAttention()
        self.fc = torch.nn.Linear(d_model * 3, 512)
 
    def forward(self, x1, x2, x3):
        q = torch.matmul(x1, self.q.in_proj_weight)
        k = torch.matmul(x2, self.k.in_proj_weight)
        v = torch.matmul(x3, self.v.in_proj_weight)
        q, k, v = self.q._split_heads(q, k, v)
        q, k, v = self.q._scaled_dot_product_attention(q, k, v)
        attn, output = self.q._attn_output(q, v)
        attn, q = self.q._merge_heads(attn, q)
        q, k, v = self.k._split_heads(k, v)
        q, k, v = self.k._scaled_dot_product_attention(q, k, v)
        attn, output = self.k._attn_output(q, v)
        attn, k = self.k._merge_heads(attn, k)
        q, k, v = self.v._split_heads(v, v)
        q, k, v = self.v._scaled_dot_product_attention(q, k, v)
        attn, output = self.v._attn_output(q, v)
        attn, v = self.v._merge_heads(attn, v)
        x = torch.cat([attn, q, k, v], dim=-1)
        x = self.fc(x)
        x = self.dropout(x)
        return x

# Initializing the model
m = Model(3, 2, 2, 0.001)

# Inputs to the model
x1 = torch.randn(1, 3, 3)
x2 = torch.randn(1, 3, 2)
x3 = torch.randn(1, 2, 3)
