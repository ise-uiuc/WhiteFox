
class Model(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, attn_mask):
        super().__init__()
        self.fc = torch.nn.Linear(384, hidden_size)
        self.self_attn = torch.nn.MultiheadAttention(hidden_size, num_heads)
        self.fc_2 = torch.nn.Linear(hidden_size, 384)
        self.attn_mask = attn_mask
 
    def forward(self, x1):
        y1 = self.fc(x1)
        y2 = self.self_attn(y1, y1, y1, self.attn_mask)
        y3 = self.fc(y2[0])
        v1 = x1 + y3
        return v1

# Initializing the model
hidden_size = 128
n_heads=4
attn_mask = torch.Tensor([[0, 1, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]).byte()
m = Model(hidden_size, n_heads, attn_mask)

# Inputs to the model
x1 = torch.randn(2, 384)
