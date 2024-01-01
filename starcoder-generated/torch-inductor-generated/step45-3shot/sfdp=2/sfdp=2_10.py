
class M(torch.nn.Module):
    def __init__(self, query, key, value, dropout_p, inv_scale_factor):
        super().__init__()

    def forward(self, q1, k1, v1):
        qk = torch.matmul(q1, k1.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(self, softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(5, 16, 64, 64)
k1 = torch.randn(5, 16, 64, 64)
v1 = torch.randn(5, 16, 64, 64)
dropout_p = 0.2
inv_scale_factor = 8.0
