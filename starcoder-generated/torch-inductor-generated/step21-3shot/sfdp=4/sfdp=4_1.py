
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 1
        self.head_dim = 1
    
    def forward(self, query, key, value, attn_mask):
        qk = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = torch.matmul(attn_weight, value)
        return output
    
# Initializing the model
num_heads = 2
head_dim = 16
m = Model(num_heads, head_dim)

# Inputs to the model
query = torch.randn(2, 2, 8, 16)
key = torch.randn(2, 2, 8, 16)
value = torch.randn(2, 2, 8, 16)
attn_mask = torch.logical_not(torch.eye(8)).unsqueeze(0).unsqueeze(0)
