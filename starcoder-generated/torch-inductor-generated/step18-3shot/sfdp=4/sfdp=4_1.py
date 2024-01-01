
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 10
        self.transformer = torch.nn.Transformer(d_model=255, nhead=self.num_heads)
 
    def forward(self, x1, x2):
        return self.transformer(x=x1, attn_mask=x2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 255, 256)
x2 = torch.randn(1, 256, 256)
