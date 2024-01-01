
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, inv_scale_factor, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        _v = torch.nn.functional.dropout(scaled_qk.softmax(dim=-1), p=dropout_p).matmul(v)
        v = v.unsqueeze(1)
        _v = _v.unsqueeze(2)
        sum_vv = torch.sum(torch.multiply(v, _v), dim=-1)
        z = torch.sum(torch.multiply(sum_vv, torch.exp(qk / 0.7).div(inv_scale_factor)), dim=-1)
        return z, qk

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 8, 5, 5)
k = torch.randn(1, 8, 5, 5)
v = torch.randn(1, 8, 5, 5)
inv_scale_factor = 1e-4
dropout_p = 0.2
