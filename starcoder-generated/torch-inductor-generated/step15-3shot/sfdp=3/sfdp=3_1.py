
class Model(torch.nn.Module):
    def forward(self, q, k, v, scale_factor, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(m.n_head, m.batch_size, m.d_k)
k = torch.randn(m.n_head, m.batch_size, m.d_k)
v = torch.randn(m.n_head, m.batch_size, m.d_v)
scale_factor = 1.0 / math.sqrt(m.d_k)
dropout_p = 0.2
