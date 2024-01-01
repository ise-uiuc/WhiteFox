
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_mask = torch.ones(8 * 2, 8 * 2)

    def forward(self, q, k, v):
        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = attn / math.sqrt(8)
        attn = attn + self.attention_mask
        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, 0.3)
        output = torch.matmul(attn, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(64 * 2, 8)
x2 = torch.randn(64 * 2, 8)
x3 = torch.randn(64 * 2, 8)
