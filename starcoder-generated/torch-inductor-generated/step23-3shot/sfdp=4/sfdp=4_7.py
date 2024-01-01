
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = torch.nn.BatchNorm2d(dim)
        self.norm2 = torch.nn.BatchNorm2d(dim)
        self.att = torch.nn.MultiheadAttention(dim, dim, num_heads)
        self.mlp = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim * mlp_ratio, 1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(dim * mlp_ratio, dim, 1, padding=0),
        )
 
    def forward(self, x):
        x = x.transpose(1, -1).contiguous()
        x = x.view(
            -1,
            x.size(1),
            x.size(2) * x.size(3),
        )
        x1, _ = self.att(x, x, x)
        x1 = x1.view(
            -1,
            x.size(1),
            x.size(2),
            x.size(3),
        )
        x1 = x1.transpose(1, -1).contiguous()
        x2 = self.mlp(x.transpose(1, -1).contiguous())
        x2 = x2.transpose(1, -1).contiguous()
        return self.norm1(x1.reshape(x.shape)) + self.norm2(x2.reshape(x.shape))

# Initializing the model
dim = 64
num_heads = 2
mlp_ratio = 1
batch_size = 1
seq_len = 32
pos_embedding = torch.randn((batch_size, seq_len, dim)).transpose(1, -1).contiguous()
x = pos_embedding.transpose(1, 2).reshape((seq_len, -1, dim))
m = Model(dim, num_heads, mlp_ratio)

# Inputs to the model
