
class Model(torch.nn.Module):
    def forward(self, q, k, v):
        _qk = torch.matmul(q, k.transpose(-2, -1))
        __inv_scale_factor__ = math.sqrt(k.shape[-1])
        __dropout_p__ = 0.9
        dropout_qk = torch.nn.functional.dropout(_qk / __inv_scale_factor__, p=__dropout_p__)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(2, 8, 16)
k = torch.randn(2, 4, 16)
v = torch.randn(2, 4, 8)
