
class Model(torch.nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout_p: float = 0.5, scale_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.scale_factor = scale_factor
        self.head_dim = int(dim / num_heads)
        self.q_layer = torch.nn.Linear(self.dim, self.dim)
        self.k_layer = torch.nn.Linear(self.dim, self.dim)
        self.v_layer = torch.nn.Linear(self.dim, self.dim)
        self.dropout_layer = torch.nn.Dropout(self.dropout_p)
        self.out_layer = torch.nn.Linear(self.dim, self.dim)

    def forward(self, x1):
        q = self.q_layer(x1)
        k = self.k_layer(x1)
        v = self.v_layer(x1)
        return self.out_layer(self.dropout_layer(self.scaled_attention(q, k, v)))

    def scaled_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        softmax_qk = torch.matmul(
            q, k.transpose(-2, -1)
        ).div(self.scale_factor).softmax(dim=-1)
        return self.dropout_layer(softmax_qk.matmul(v))

# Initializing the model
m = Model(dim=10, num_heads=5, dropout_p=0.5, scale_factor=2.0)

# Input to the model
x1 = torch.randn(2, 5, 10)

# Output of the model
