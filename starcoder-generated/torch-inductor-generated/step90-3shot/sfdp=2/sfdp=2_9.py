
class Model(torch.nn.Module):
    def __init__(self, dim, inv_scale_factor=1e-4, dropout_p=0):
        super().__init__()
        self.dim = dim
        self.w = torch.nn.Linear(dim, dim)
        self.dropout = torch.nn.Dropout(dropout_p)
        inv_scale_factor = torch.tensor([inv_scale_factor])
 
    def forward(self, query, key, value):
        q = self.w(query)
        k, v = key.transpose(-2, -1), value.transpose(-2, -1)
        qk = torch.matmul(q, k)
        scaled_qk = qk.div(self.dim**-0.5 * inv_scale_factor**0.5)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk.transpose(-2, -1))
        output = torch.matmul(value, dropout_qk)
        return output, softmax_qk

# Initializing the model
m = Model(8)

# Inputs to the model
query = torch.randn(1, 3, 2)
key = torch.randn(8, query.shape[-1], 2)
value = torch.randn(8, query.shape[-2], 2)
__output__, __softmax_qk__ = m(query, key, value)

