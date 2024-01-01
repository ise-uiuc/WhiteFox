
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, output_dim, scale_factor=1. / np.sqrt(128), dropout=0.5):
        super().__init__()
        self.scale_factor = torch.nn.Parameter(torch.full((1,), scale_factor))
        self.dropout = dropout
        self.dropout_p = torch.nn.Parameter(torch.full((1,), dropout))
        self.query_norm = torch.nn.LayerNorm((1, 1, query_dim))
        self.key_norm = torch.nn.LayerNorm((1, 1, key_dim))
        self.value_norm = torch.nn.LayerNorm((1, 1, value_dim))
        self.weight = torch.nn.Parameter(torch.zeros(value_dim, key_dim))
    
    def forward(self, x1, x2):
        Q = self.query_norm(x1)
        K = self.key_norm(x2)
        V = self.value_norm(x2)
        Q_ = Q.reshape(-1, Q.size(-1)).transpose(0, 1)
        K_ = K.reshape(-1, K.size(-1))
        V_ = V.reshape(-1, V.size(-1))
        qk = torch.matmul(Q_, K_)
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = torch.nn.functional.softmax(scaled_qk, -1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, self.dropout_p, True)
        o = dropout_qk.matmul(V_)
        o = o.reshape(-1, 1, 1, o.size(-1))
        o = o.transpose(0, -1)
        w = self.weight.transpose(-2, -1)
        v5 = o.matmul(w)
        return v5

# Initializing the model
m = Model(256, 256, 256, 256)

# Inputs to the model
x1 = torch.randn(1, 32, 256)
x2 = torch.randn(1, 256, 256)
