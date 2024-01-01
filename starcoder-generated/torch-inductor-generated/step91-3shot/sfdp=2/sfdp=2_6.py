
q = torch.randn(8, 32, 128)
k = torch.randn(8, 20, 128)
v = torch.randn(8, 20, 512)
m = torch.randn(8, 32, 512)
inv_scale_factor = torch.randn(8, 1, 1)
dropout_p = 0.5

def compute_attention(query, key, value, dropout_p=0.5):
    qk = torch.matmul(query, key.transpose(-2, -1))
    scaled_qk = qk.div(inv_scale_factor)
    softmax_qk = scaled_qk.softmax(dim=-1)
    dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
    output = dropout_qk.matmul(value)
    return output

