
class Model(torch.nn.Module):
    def __init__(self, q, k, v, n_heads, dropout):
        super().__init__()
        self.q, self.k, self.v = q, k, v
        self.n_heads, self.dropout = n_heads, dropout
 
    def forward(self, x1):
        q = self.q(x1)
        k, v_origin = self.k(x1), self.v(x1)
        v = v_origin.softmax(dim=1)
        scaled_qk = torch.matmul(q, (k.transpose(1, 0))) / math.sqrt(self.n_heads)
        softmax_qk = scaled_qk.softmax(dim=0)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, self.dropout)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
q, k, v = [torch.nn.Linear(4, 3) for i in range(3)]
n_heads, dropout = 2, 0.0
m = Model(q, k, v, n_heads, dropout)

# Inputs to the model
x1 = torch.randn(1, 2, 3, 4)
