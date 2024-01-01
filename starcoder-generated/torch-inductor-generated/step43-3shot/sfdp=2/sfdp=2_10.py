
class Model(torch.nn.Module):
    def __init__(self, num_qkv, num_heads, dropout_p, inv_scale_factor):
        super().__init__()
        self.query = torch.nn.Dense(num_qkv, num_qkv)
        self.key = torch.nn.Dense(num_qkv, num_qkv)
        self.value = torch.nn.Dense(num_qkv, num_qkv)
        self.inv_scale_factor = inv_scale_factor
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, x1):
        q = self.query(x1)
        k = self.key(x1)
        v = self.value(x1)
 
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(num_qkv=10, num_heads=3, dropout_p=0.1, inv_scale_factor=1.0)

# Inputs to the model
x1 = torch.randn(1, 10)
