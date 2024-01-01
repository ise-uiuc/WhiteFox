
class Model(torch.nn.Module):
    def __init__(self, dim, dropout_p):
        super().__init__()
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.inv_scale_factor = nn.Parameter(torch.tensor(1.))
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value):
        q = self.q(query)
        k = self.k(key)
        v = value.transpose(-2, -1)
        qk = torch.matmul(q, k)
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(dim, dropout_p)

# Inputs to the model
query = torch.randn(1, 8, 64, 64)
key = torch.randn(1, 8, 128, 128)
value = torch.randn(1, 8, 128, 128)
