
def scaled_dot_product_attention(query, key, value, inv_scale):
    w = torch.matmul(query, key.transpose(-2, -1))
    w = w / inv_scale
    w = w.softmax(dim=-1)
    output = torch.matmul(w, value)
    return output

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.fc_q = torch.nn.Linear(d_model, d_model)
        self.fc_k = torch.nn.Linear(d_model, d_model)
        self.fc_v = torch.nn.Linear(d_model, d_model)
        self.fc_o = torch.nn.Linear(d_model, d_model)

    def attention(self, query, key, value):
        inv_scale = 1. / (self.d_k ** 0.5)
        attention = scaled_dot_product_attention(query, key, value, inv_scale)
        return attention

    def forward(self, q, k, v):
        q = self.fc_q(q).view(q.shape[0], -1, self.num_heads, self.d_k)
        k = self.fc_k(k).view(q.shape[0], -1, self.num_heads, self.d_k)
        v = self.fc_v(v).view(q.shape[0], -1, self.num_heads, self.d_k)
        q = q.permute(2, 0, 1, 3)
        k = k.permute(2, 0, 1, 3)
        v = v.permute(2, 0, 1, 3)
        a = self.attention(q, k, v)
        a = a.permute(1, 2, 0, 3).contiguous().view(a.shape[1], -1, a.shape[3])
        o = self.fc_o(a)

        return o

class Model(torch.nn.Module):
    def __init__(self, d_model=768, num_heads=8):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)

    def forward(self, x1, x2):
        y = self.mha(x1, x2, x2)
        return y

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 16, 768)
x2 = torch.randn(2, 16, 768)
