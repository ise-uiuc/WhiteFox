
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, dropout_p=0.5, inv_scale_factor=32768):
        super().__init__()
        self.scale_factor = inv_scale_factor
        
        self.query = torch.nn.Linear(query_dim, key_dim)
        self.key = torch.nn.Linear(key_dim, key_dim)
        self.value = torch.nn.Linear(value_dim, key_dim)
        self.softmax = torch.nn.Softmax(-1)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, t1, t2):
        qkv = []
        for t in [t1, t2]:
            qkv.append(self.key(self.query(t)))
        q = torch.stack(qkv).sum(0)
        k = torch.stack(qkv).sum(0)
        qk = torch.matmul(q, k.transpose(-2, -1) / self.scale_factor)
        qk /= self.scale_factor
        softmax_qk = self.softmax(qk)
        dropout_qk = self.dropout(softmax_qk)
        output = self.value(dropout_qk.matmul(torch.stack(qkv).sum(0)))
        return output

# Initializing the model
m = Model(8, 8, 8)

# Inputs to the model
t1 = torch.randn(1, 8, 16)
t2 = torch.randn(1, 8, 32)
