
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_dim = 64
        self.dropout_p = 0

    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.emb_dim ** -0.5)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 2, 64)
k = torch.randn(1, 4, 64)
v = torch.randn(4, 2, 64)
