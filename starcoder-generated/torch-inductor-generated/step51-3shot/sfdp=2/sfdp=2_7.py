
class Model(torch.nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout_p):
        super().__init__()
        self.head_dim = hidden_dim // num_heads
 
        self.query = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
 
        self.dropout_p = dropout_p
 
    def forward(self, q, v):
        q = self.query(q)
        k = self.key(v)
        v = self.value(v)
 
        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(-3, -2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(-3, -2)
        v = v.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(-3, -2)
 
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = 1. / math.sqrt(q.size(-1))
        scaled_qk = qk.div(inv_scale_factor)
 
        dropout_qk = torch.nn.functional.dropout(scaled_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
 
        return output

# Initializing the model
m = Model(8, 128, 0.1)

# Inputs to the model
q = torch.randn(3, 64, 128)
v = torch.randn(6, 64, 128)
