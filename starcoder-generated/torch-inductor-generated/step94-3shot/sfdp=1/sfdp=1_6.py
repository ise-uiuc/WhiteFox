
class Model(torch.nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads=8, dropout_p=0.1):
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
 
        self.scale_factor = (q_dim // num_heads) ** 0.5
        self.fc = torch.nn.Linear(q_dim, kv_dim * 3)
 
    def forward(self, query, key, value):
        qkv = self.fc(query)
        query, key, value = torch.split(qkv, [self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        query = query.view(query.size(0), self.num_heads, query.size(1)//self.num_heads, query.size(2))
        key = key.view(key.size(0), self.num_heads, key.size(1) // self.num_heads, key.size(2))
        value = value.view(value.size(0), self.num_heads, value.size(1) // self.num_heads, value.size(2))
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1).contiguous()
        value = value.permute(0, 2, 1, 3).contiguous()
 
        qk = torch.matmul(query, key)
        scaled_qk = qk.div(self.scale_factor).float()
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(output.size(0), output.size(1), output.size(2)*output.size(3))
        return output

# Initializing the model
m = Model(1024, 2048, 4, 0.1)

# Inputs to the model
q = torch.randn(1, 4096, 1024)
k = torch.randn(1, 32, 2048)
v = torch.randn(1, 32, 2048)
