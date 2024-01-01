
class Model(torch.nn.Module):
    def __init__(self, input_size=64, hidden_size=64, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.q = torch.nn.Parameter(torch.zeros(hidden_size, input_size))
        self.k = torch.nn.Parameter(torch.zeros(hidden_size, input_size))
        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        torch.nn.init.normal_(self.q)
        torch.nn.init.normal_(self.k)

    def forward(self, x1, x2):
        q = self.layer_norm(self.q)
        k = self.layer_norm(self.k)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = 1/math.sqrt(k.size(-1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout)
        output = dropout_qk.matmul(x2)
        return output

# Initializing the model
m = Model(64, 64)

# Inputs to the model
x1 = torch.randn(2, 64, 64)
x2 = torch.randn(2, 64, 64)
__o__ = m(x1, x2)

