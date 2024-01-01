
class Model(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.q = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v = torch.nn.Linear(hidden_dim, hidden_dim)
        self.inv_scale_factor = hidden_dim ** -0.5
        self.dropout_p = 0.5
    
    def forward(self, q, k, v):
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(hidden_dim)

# Inputs to the model
q = torch.randn(8, 1, 384)
k = torch.randn(8, 1, 384)
v = torch.randn(8, 1, 384)
