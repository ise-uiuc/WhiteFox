
class Model(torch.nn.Module):
    def __init__(self, num_heads, dim_qk, dim_v, dropout_p, scale_factor=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.dropout_p = dropout_p
        self.scale_factor = scale_factor

        self.project_q = torch.nn.Linear(self.dim_qk, self.num_heads * self.dim_qk)
        self.project_k = torch.nn.Linear(self.dim_qk, self.num_heads * self.dim_qk)
        self.project_v = torch.nn.Linear(self.dim_v, self.num_heads * self.dim_v)

    def forward(self, query, key, value):
        q = self.project_q(query).chunk(self.num_heads, dim=-1)
        k = self.project_k(key).chunk(self.num_heads, dim=-1)
        v = self.project_v(value).chunk(self.num_heads, dim=-1)

        q = torch.cat(q, dim=0)
        k = torch.cat(k, dim=0)
        v = torch.cat(v, dim=0)

        q = q.reshape_as(k)
        k = k.reshape_as(v)

        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = F.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        
        return output
    
# Initializing the model
num_heads = 2
dim_qk = 10
dim_v = 10
dropout_p = 0.1
m = Model(num_heads, dim_qk, dim_v, dropout_p)

# Inputs to the model
query = torch.randn(4, 2, num_heads, dim_qk)
key = torch.randn(4, 2, num_heads, dim_qk)
value = torch.randn(4, 2, num_heads, dim_v)
