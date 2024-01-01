
class Model(torch.nn.Module):
    def __init__(self, natt, ninp):
        super().__init__()
        # Attention mask
        self.register_buffer("am", torch.zeros((1, natt, 1, 1), dtype=torch.bool))
 
    def forward(self, x1, x2):
        attn = x1 @ x2.transpose(-2, -1) / math.sqrt(x1.shape[-1])
        attn = attn + self.am
        attn = torch.softmax(attn, dim=-1)
        output = attn @ x2
        return output

# Initializing the model
m = Model(4, 16)

# Inputs to the model
x1 = torch.randn(1, 4, 8, 16)
