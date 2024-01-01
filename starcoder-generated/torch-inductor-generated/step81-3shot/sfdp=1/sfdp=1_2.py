
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(1 / math.sqrt(k.size(-1)))
        softmax_qk = F.softmax(scaled_qk, dim=-1)
        dropout_qk = F.dropout(softmax_qk, p=0.2)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(128, 64, 32)
k = torch.randn(128, 64, 48)
v = torch.randn(128, 64, 48)
__output__, __attn_weights__ = m(q, k, v)

