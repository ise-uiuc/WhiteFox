
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_key: int, d_value: int, d_model: int):
        super().__init__()
        self.scale = torch.sqrt(torch.FloatTensor([d_key]))
        self.d_key = d_key

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(query, key.transpose(-2, -1))
        x = x / self.scale
        x = F.softmax(x, dim=-1)
        y = torch.matmul(x, value)
        return y
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.attention = ScaledDotProductAttention(8, 8, 8)
    
    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        x2 = self.conv(x1)
        y = self.attention(x2, x2, x2)
        return y

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 3, 64, 64)
