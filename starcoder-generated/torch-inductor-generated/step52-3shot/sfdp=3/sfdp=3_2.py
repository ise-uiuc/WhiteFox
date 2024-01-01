
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, dropout_p):
        super().__init__()
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, query, key, value, scale_factor=1):
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        q = torch.transpose(q, -2, -1)
        k = torch.transpose(k, -2, -1)
        qk = torch.matmul(q, k)
        scaled_qk = qk * scale_factor
        softmax_qkl = self.softmax(scaled_qk)
        dropout_qkl = self.dropout(softmax_qkl)
        output = torch.matmul(dropout_qkl, v)
        output = torch.transpose(output, -2, -1)
        return output

# Initializing the model
m = Model(dim=512, num_heads=4, dropout_p=0.1)

# Inputs to the model
x1 = torch.randn(1, 50, 512)
x2 = torch.randn(1, 60, 512)
x3 = torch.randn(1, 60, 512)
