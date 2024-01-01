
def scaled_dot_product_attention(q, k, v, scale_factor, dropout_p):
    qk = torch.matmul(q, k.transpose(-2, -1))
    scaled_qk = qk.mul(scale_factor)
    softmax_qk = scaled_qk.softmax(dim=-1)
    dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
    output = dropout_qk.matmul(v)
    return output

class MHA(torch.nn.Module):
    def __init__(self, d_model, dropout_p=0.1):
        super().__init__()
        self.dropout_p = dropout_p
        self.query_linear = torch.nn.Linear(d_model, d_model)
        self.key_linear = torch.nn.Linear(d_model, d_model)
        self.value_linear = torch.nn.Linear(d_model, d_model)
        self.scale_factor = torch.sqrt(torch.FloatTensor([d_model])).to(device='cpu')
 
    def forward(self, q, k, v):
        q = self.query_linear(q)
        k = self.key_linear(k)
        v = self.value_linear(v)
        attention = scaled_dot_product_attention(q, k, v, self.scale_factor, self.dropout_p)
        return attention

class Model(torch.nn.Module):
    def __init__(self, d1):
        super().__init__()
        self.mh_attention = MHA(d1)
 
    def forward(self, x1, x2):
        v1 = self.mh_attention(x1, x2, x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6
```

# Initializing the model
d1 = 4
m = Model(d1)

# Inputs to the model
x1 = torch.randn(1, d1, 64, 64)
x2 = torch.randn(1, d1, 128, 128)
