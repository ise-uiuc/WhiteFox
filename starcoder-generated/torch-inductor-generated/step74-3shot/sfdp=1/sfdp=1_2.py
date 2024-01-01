
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        inv_scale_factor = 1 / np.sqrt(x1.shape[-1])
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        batch_size = softmax_qk.shape[0]
        num_heads = softmax_qk.shape[-1]
        dropout_p = 0.1
        dropout_q = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        value = x2
        dropout_q = dropout_q.reshape(1, batch_size * num_heads, -1)
        value = value.reshape(1, batch_size * num_heads, -1)
        output = dropout_q.matmul(value)
        output = output.reshape(batch_size, num_heads, -1)
        return output
 
# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(1, 8, 64)
x2 = torch.randn(1, 8, 64)
