
class Model(torch.nn.Module):
    def forward(self, q, k, v, dropout_p):
        inv_scale_factor = 1.0 / np.sqrt(q.shape[-1])
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 3, 8, 256)
k = torch.randn(1, 3, 8, 256)
v = torch.randn(1, 3, 8, 256)
dropout_p = 0.2
