
class Model(torch.nn.Module):
    def forward(self, x):
        v1 = torch.softmax((x @ y.transpose(-2, -1) / math.sqrt(x.size(-1))) + attn_mask, dim=-1)
        z = v1 @ x
        return z

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(16, 8, 64)
y = torch.randn(16, 9, 64)
attn_mask = torch.randint(0, 2, (16, 8, 8), dtype=torch.long)
