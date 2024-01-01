
model = M.Module()
model.query = torch.nn.Linear()
model.key = torch.nn.Linear()
model.value = torch.nn.Linear()

def scaled_dot_product_attention(query, key, value, mask=None, scale_factor=np.sqrt(dk)):
    dk = query.size(-1)
    qk = torch.matmul(query, key.transpose(-2, -1))
    scaled_qk = qk * scale_factor
    softmax_qk = scaled_qk.softmax(dim=-1)
    dropout_qk = torch.nn.functional.dropout(softmax_qk, p = 0)
    output = dropout_qk.matmul(value)
    return output

def mlp(x1):
    x3 = torch.tanh(model.l1(x1))
    x4 = torch.tanh(model.l2(x3))
    return x4

def forward(x1, x2):
    x7 = scaled_dot_product_attention(mlp(x1), mlp(x2), mlp(x3))
    x8 = scaled_dot_product_attention(mlp(x4), mlp(x2), mlp(x3))
    return x7 + x8

# Input
output = forward(x1, x2)

