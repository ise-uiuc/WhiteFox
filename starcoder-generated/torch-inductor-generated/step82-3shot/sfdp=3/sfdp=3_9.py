
class Model(torch.nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.dim = d_model
        self.dropout = torch.nn.Dropout(0.4)
        self.linear = torch.nn.Linear(d_model, d_model)
        self.q = torch.nn.Linear(d_model, d_model)
        self.k = torch.nn.Linear(d_model, d_model)
        self.v = torch.nn.Linear(d_model, d_model)
 
    def forward(self, q, k, v):
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
 
        q = q.view(q.size(0), q.size(1), self.num_heads, self.dim // self.num_heads)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.dim // self.num_heads)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.dim // self.num_heads)
 
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
 
        attention = (q @ k.transpose(-2, -1)) * (q.size(-1) ** (-0.5))
 
        attention = torch.mul(attention.permute(2, 0, 1, 3), 10000.0)
 
        attention /= attention.sum((-2, -1), keepdim=True)
 
        attention = torch.matmul(attention, value)
 
        attention = attention.transpose(1, 2).contiguous()
 
        output = attention.view(q.size(0), q.size(1), -1)
 
        output = self.linear(output)
 
        output = self.dropout(output)
 
        return output
 
 
model = Model(num_heads=8, d_model=128)

# Inputs to the model
q = torch.randn(64, 8, 128)
k = torch.randn(64, 8, 128)
v = torch.randn(64, 8, 128)
