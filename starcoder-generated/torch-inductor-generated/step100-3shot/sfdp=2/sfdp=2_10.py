
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, nheads, dropout_p):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.nheads = nheads
        self.dropout_p = dropout_p

    def forward(self, query, key, value, inv_scale_factor):
        _ = query.size()
        _ = key.size()
        _ = value.size()

        qk = torch.matmul(query, key.transpose(-2, -1))

        scaled_qk = qk.div(inv_scale_factor)

        softmax_qk = scaled_qk.softmax(dim=-1)

        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)

        output = dropout_qk.matmul(value)

        return output

# Initializing the model
q, k, v, isf = 512, 512, 3, 256
m = Model(q, k, nheads=8, dropout_p=0.3)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 32, 64, 64)
x3 = torch.randn(1, 32, 128, 128)
