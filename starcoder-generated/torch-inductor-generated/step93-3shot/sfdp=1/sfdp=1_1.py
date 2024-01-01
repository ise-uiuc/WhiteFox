
class Model(torch.nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float):
        super(Model, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.qk_scale = torch.sqrt(embedding_dim // num_heads)
        self.dropout_p = dropout

        self.conv1d3 = torch.nn.Conv1d(self.embedding_dim, self.embedding_dim, 3)
        self.norm1d1 = torch.nn.LayerNorm([self.embedding_dim])

        self.conv1d4 = torch.nn.Conv1d(self.embedding_dim, self.embedding_dim, 4)
        self.norm1d2 = torch.nn.LayerNorm([self.embedding_dim])

        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor
    ) -> torch.Tensor:
        def forward1(embed: torch.Tensor) -> torch.Tensor:
            embed1 = self.conv1d3(embed)
            embed1 = self.norm1d1(embed1)
            embed1 = torch.relu(embed1)

            embed2 = self.conv1d4(embed1)
            embed2 = self.norm1d2(embed2)
            embed2 = torch.relu(embed2)

            embed = torch.cat((embed1, embed2), dim=2)
            embed = torch.transpose(embed, 2, 1)

            return embed

        def forward2(embed: torch.Tensor) -> torch.Tensor:
            embed = self.linear(embed)
            return embed

        src = forward1(x1) + forward1(x2) + forward1(x3)

        # Apply multi-head attention as described below
        attn = torch.matmul(src, src.transpose(-2, -1))
        attn = attn / self.qk_scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, src)

        out = forward2(out)

        return out

# Initializing the model
m = Model(16, 8, 0.2)
m = m.to("cuda")

# Inputs to the model
x1 = torch.randn(1, 1, 16, 64)
x2 = torch.randn(1, 1, 16, 64)
x3 = torch.randn(1, 1, 16, 64)
x1 = x1.to("cuda")
x2 = x2.to("cuda")
x3 = x3.to("cuda")
