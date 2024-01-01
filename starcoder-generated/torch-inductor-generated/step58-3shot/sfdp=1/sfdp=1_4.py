
class Model(torch.nn.Module):
    def __init__(self):
        self.embed_dim = 64
        self.num_heads = 4
        self.head_dim = 64
        self.dropout_p = 0.3
        self.inv_scale_factor = 1.0 / np.sqrt(self.head_dim)
        super().__init__()
        self.wQ = torch.nn.Linear(self.embed_dim,
                                  self.embed_dim)
        self.wK = torch.nn.Linear(self.embed_dim,
                                  self.embed_dim)
        self.wV = torch.nn.Linear(self.embed_dim,
                                  self.embed_dim)
 
    def forward(self, q, k, v):
        B, N, C = v.shape
        h_drop = (torch.nn.functional.dropout(q, p=self.dropout_p),
                  torch.nn.functional.dropout(k, p=self.dropout_p),
                  torch.nn.functional.dropout(v, p=self.dropout_p))
        q = self.wQ(h_drop[0])
        k = self.wK(h_drop[1])
        v = self.wV(h_drop[2])
        q = q.reshape((B, N, self.num_heads, self.head_dim))
        k = k.reshape((B, N, self.num_heads, self.head_dim))
        v = v.reshape((B, N, self.num_heads, self.embed_dim //
                        self.num_heads))
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div( self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 128, 64)
x2 = torch.randn(128, 64)
x3 = torch.randn(128, 48)
