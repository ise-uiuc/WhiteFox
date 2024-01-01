
def model(query, key, value, inv_scale_factor=1. / np.sqrt(64), dropout_p=0.):
    qk = torch.matmul(query, key.transpose(-2, -1))
    scaled_qk = qk.div(inv_scale_factor)
    softmax_qk = torch.nn.functional.softmax(scaled_qk, -1)
    dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
    return torch.matmul(dropout_qk, value)

# Initializing the model
query = torch.randn(1, 8, 64, 64)
value = torch.randn(1, 8, 64, 64)
key = torch.randn(1, 8, 64, 64)
