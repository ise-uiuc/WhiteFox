
class Model1(torch.nn.Module):
    def __init__(self, dim=64, depth=None, seq_len=None):
        super().__init__()
        self.dim = dim
        self.pos_emb = RelativePositionEmbedding(dim)
        self.embed = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, dim)
        )
        self.blocks = torch.nn.Sequential(
            *[TransformerBlock(dim) for _ in range(depth)],
            LayerNormAfterResidual(dim)
        )
 
    def forward(self, x):
        x = self.pos_emb(x)
        x = self.embed(x)
        x = x.transpose(-1, -2)
        x = self.blocks(x)
        return x


# Initializing the model
m = Model(depth=1, seq_len=6)

# Inputs to the model
x1 = torch.randn(1, 1, m.dim, m.dim)
