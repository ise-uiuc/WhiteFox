
class Model(torch.nn.Module):
    def __init__(self, q_n_hidden, k_n_hidden, v_n_hidden, scale_factor, dropout_p):
        super().__init__()
        self.q_proj = torch.nn.Linear(q_n_hidden, k_n_hidden, bias=False)
        self.v_proj = torch.nn.Linear(v_n_hidden, k_n_hidden, bias=False)
        self.k_proj = torch.nn.Linear(k_n_hidden, k_n_hidden, bias=False)
        self.softmax_d = -1
        self.dropout_p = dropout_p
        self.scale_factor = scale_factor
    
    def forward(self, q, k, v):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        x1 = torch.matmul(q, k.transpose(-2, -1))
        x2 = x1.div(self.scale_factor)
        x3 = F.softmax(x2, dim=self.softmax_d)
        x4 = F.dropout(x3, p=self.dropout_p)
        return torch.matmul(x4, v)

# Initializing the model
m = Model(q_n_hidden=2048, k_n_hidden=2048, v_n_hidden=2048, scale_factor=float(1./np.sqrt(2048)), dropout_p=0.0)

# Inputs to the model
x1 = torch.randn(3072, 2048)
x2 = torch.randn(3072, 2048)
x3 = torch.randn(3072, 2048)
