
class Model(torch.nn.Module):
    def __init__(self, nb_head, nb_qkvi_dim, nb_v_dim, dropout_p=0.0):
        super().__init__()
        self.nb_head = nb_head
        self.nb_qkvi_dim = nb_qkvi_dim
        self.nb_v_dim = nb_v_dim
        self.dropout_p = dropout_p
        self.qkV_scale = 1 / math.sqrt(nb_qkvi_dim)
        self.q_proj = torch.nn.Linear(nb_qkvi_dim, nb_qkvi_dim, bias=False)
        self.k_proj = torch.nn.Linear(nb_qkvi_dim, nb_qkvi_dim, bias=False)
        self.v_proj = torch.nn.Linear(nb_qkvi_dim, nb_v_dim, bias=False)
        self.drop = torch.nn.Dropout(dropout_p)
 
    def forward(self, x2, x3, x4):
        qk = torch.matmul(self.q_proj(x2), self.k_proj(x3).transpose(-2, -1))
        scaled_qk = qk.div(self.qkV_scale)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.drop(softmax_qk)
        output = dropout_qk.matmul(self.v_proj(x4))
        return output

# Initializing the model
m = Model(8, 1024, 1024, 0.0)

# Inputs to the model
x2 = torch.randn(64, 8, 1024, 1)
x3 = torch.randn(64, 16, 1024, 1)
x4 = torch.randn(64, 1024, 1024, 1)
