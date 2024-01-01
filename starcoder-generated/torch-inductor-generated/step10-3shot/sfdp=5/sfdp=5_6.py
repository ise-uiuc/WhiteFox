
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = x1.size(0)
        v2 = x1.size(1)
        v3 = x1.size(2)
        v4 = x1.size(3)
        v5 = torch.matmul(x1.view(v1, v2, v3 * v4), x1.view(v2, v1, v3 * v4).transpose(1, 2))
        v5 = v5 / math.sqrt(v3 * v4)
        v6 = v5 + attn_mask
        v7 = torch.softmax(v6, dim=-1)
        y__ = torch.matmul(v7, x1)    
        return y__

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(batch_size, hidden_size, seq_length, seq_length)
