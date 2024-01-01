
class Model(torch.nn.Module):
    def __init__(self, n_head, hidden_dim, dropout_p):
        super().__init__()
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.scale_factor = hidden_dim ** -0.5
        self.key = torch.nn.Linear(hidden_dim, hidden_dim)
        self.value = torch.nn.Linear(hidden_dim, hidden_dim)
        self.query = torch.nn.Linear(hidden_dim, hidden_dim)
 
    def forward(self, q0, k0, v0):
        q = self.query(q0)
        k = self.key(k0)
        v = self.value(v0)
        qh = q.reshape(q.size(0), -1, self.n_head, self.hidden_dim // self.n_head).transpose(1, 2)
        kh = k.reshape(k.size(0), -1, self.n_head, self.hidden_dim // self.n_head).transpose(1, 2)
        vh = v.reshape(v.size(0), -1, self.n_head, self.hidden_dim // self.n_head).transpose(1, 2)
        qk = torch.matmul(qh, kh.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p, training=self.training)
        output = dropout_qk.matmul(vh)
        output = output.transpose(1, 2).contiguous().reshape(q0.size(0), -1, self.hidden_dim)
        return output
 
# Initializing the model
n_head = 4
hidden_dim = 8
dropout_p = 0.3
m = Model(n_head, hidden_dim, dropout_p)
 
# Inputs to the model
q = torch.randn(1, 4, 64)
k = torch.randn(2, 4, 64)
v = torch.randn(2, 4, 64)
output = m(q, k, v)

