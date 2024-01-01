
def attention(x1, x2, x3, scale_factor, dropout_p):
    qk = torch.matmul(x1, x2.transpose(-2, -1))
    scaled_qk = qk.div(scale_factor)
    softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1).type(torch.float16)
    dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
    output = dropout_qk.matmul(x3)
    return output

# Initial a scale_factor and a dropout_p
scale_factor = np.random.rand(32)
dropout_p = 0.11871325310848029 * random.random()

# Inputs to the model
x1 = torch.randn(32, 2, 2, 8).type(torch.float16)
x2 = torch.randn(2, 8, 3).type(torch.float16)
x3 = torch.randn(3, 1).type(torch.float16)
