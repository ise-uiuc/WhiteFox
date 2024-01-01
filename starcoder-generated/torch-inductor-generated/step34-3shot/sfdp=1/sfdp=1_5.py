
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
 
    def forward(self, q, k, v, inv_scale_factor, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
batch_size = 1
seq_len = 8
m = Model()

# Inputs to the model
query = torch.randn(batch_size, seq_len, 16)
key = torch.randn(batch_size, seq_len, 16)
value = torch.randn(batch_size, seq_len, 16)
inv_scale_factor = torch.rand(1) * 1e-1
dropout_p = torch.__version__[0] >= '1'
