
class Model(torch.nn.Module):
    def __init__(self, dim_out, num_heads=8, dropout_p=0.1):
        super().__init__()
        self.num_heads = num_heads
 
        self.scale_factor = np.sqrt(1. / dim_out)
        self.inv_scale_factor = 1. / self.scale_factor
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_p)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        s_qk = self.softmax(scaled_qk)
        d_o = self.dropout(s_qk)
        res = d_o.matmul(value)
        return res

# Initializing the model
m = Model(64, num_heads=8, dropout_p=0.2)

# Inputs to the model
x1 = torch.randn(1, 64, 16)
x2 = torch.randn(1, 64, 32)
x3 = torch.randn(1, 64, 32)
