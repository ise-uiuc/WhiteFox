
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, scale_factor, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
dropout_p = 0.5
q = torch.randn(1, 8, 16, 32)
k = torch.randn(1, 8, 16, 32)
v = torch.randn(1, 8, 16, 32)
v1 = torch.nn.Parameter(torch.tensor(2.134503, dtype=torch.float32))
m = Model()
