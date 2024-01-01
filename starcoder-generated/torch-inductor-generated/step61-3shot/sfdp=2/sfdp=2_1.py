
class Model(torch.nn.Module):
    def __init__(self, num_heads, max_length=4):
        super().__init__()
        self.qkv = torch.nn.Linear(max_length, max_length)
        for p in self.qkv.parameters():
            p.requires_grad = False
        self.num_heads = num_heads
        self.max_length = max_length

    def forward(self, query, key, value, dropout_p=0):
        k = self.qkv(key)
        q = self.qkv(query)
        v = self.qkv(value)
        q = q.split(self.max_length // self.num_heads, dim=-1)
        k = k.split(self.max_length // self.num_heads, dim=-1)
        v = v.split(self.max_length // self.num_heads, dim=-1)
        q = torch.cat(q, dim = 0)
        k = torch.cat(k, dim = 0)
        v = torch.cat(v, dim = 0)        
        q *= self.max_length ** -0.5
        qk = torch.matmul(q, k.transpose(-2, -1))
        dropout_qk = torch.nn.functional.dropout(qk.softmax(dim=-1), p=dropout_p)
        output = dropout_qk.matmul(v)
        output = output.split(self.num_heads, dim=0)
        output = torch.cat(output, dim=-1)
        return output

# Initializing the model
m = Model(num_heads=3)

# Inputs to the model
query = torch.randn(6, 8//3, 2, 3)
key = torch.randn(4, 8//3, 3, 3)
value = torch.randn(4, 8//3, 3, 3)
