
class Model(torch.nn.Module):
    def __init__(self, q, k, v, dropout_p):
        super().__init__()
        self.q = q
        self.k = k
        self.v = v
        self.dropout_p = dropout_p
 
    def forward(self, query):
        qk = torch.matmul(query, self.k.transpose(-2, -1))
        inv_scale_factor = torch.rsqrt(torch.tensor(self.q.size()[-1], dtype=torch.float))
        softmax_qk = torch.nn.functional.softmax(qk.div(inv_scale_factor), dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(self.v)
        return output

# Initializing the model
m = Model(torch.randn(1, 3, 25, 2), torch.randn(1, 3, 25, 3), torch.randn(1, 3, 25, 3), 0.125)

# Inputs to the model
query = torch.randn(1, 3, 2, 64)
