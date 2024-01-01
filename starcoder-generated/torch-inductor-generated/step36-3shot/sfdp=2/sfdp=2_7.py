
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(3, 6)
        self.attn = torch.nn.Linear(6, 3)
 
    def forward(self, x):
        v = self.qkv(x)
        q, k, v = v.split(split_size, dim=2)
        scale_factor = k.shape[-1] ** 0.5
        q, k, v = q * scale_factor, k * scale_factor, v * scale_factor
        attn = self.attn(torch.matmul(q, k.transpose(1, 2)))
        attn = torch.softmax(attn, dim=-1)
        return attn

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10, 3)
