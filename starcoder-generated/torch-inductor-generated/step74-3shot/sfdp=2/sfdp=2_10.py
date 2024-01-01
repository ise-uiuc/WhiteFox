
class Model(torch.nn.Module):
    def __init__(self, query_channels, key_channels, output_channels, kernel_size=1, scale_factor=1, dropout_p=0):
        super().__init__()
        self.fc_kv = torch.nn.Linear(key_channels, output_channels * 2, bias=False)
        self.fc_q = torch.nn.Linear(query_channels, output_channels, bias=False)
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p
 
    def forward(self, x1, x2):
        f1 = self.fc_kv(x2)
        kv = f1.reshape(-1, x2.shape[1], 2, f1.shape[-1])
        q = self.fc_q(x1)
        qk = torch.matmul(q, kv.transpose(-2, -1))
        s = qk.div(self.scale_factor)
        m = nn.Softmax2d()
        s_ = m(s)
        d = s_.dropout(p=self.dropout_p)
        o = d.matmul(kv)
        return o

# Initializing the model
m = Model(3, 3, 4, kernel_size=1, scale_factor=2, dropout_p=0)

# Inputs to the model
x1 = torch.randn(4, 3)
x2 = torch.randn(4, 3)
