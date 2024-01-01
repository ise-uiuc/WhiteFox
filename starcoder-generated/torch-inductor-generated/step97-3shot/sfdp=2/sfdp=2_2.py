
class SoftmaxAttention(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.soft_k = nn.Linear(channels, channels).cuda()
        self.soft_v = nn.Linear(channels, channels).cuda()
        self.query = nn.Parameter(torch.randn(1, heads, 512, channels // heads)).cuda()
        self.dropout = nn.Dropout(0.1)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        a = self.soft_k(v).transpose(-2, -1)
        b = self.soft_v(v)
        c = torch.matmul(self.query, b.transpose(-2, -1)) # [1, heads, 512, 1024]
        d = c.div(0.2) # [1, heads, 512, 1024]
        e = F.softmax(d, dim=2).cuda()
        f = self.dropout(e.cuda())
        g = f.matmul(b)
        h = torch.matmul(self.query.permute(0, 1, 3, 2), g).permute(0, 2, 3, 1) 
        return h.contiguous().view(h.size(0), h.size(1), -1)

# Initializing the model
m = SoftmaxAttention(channels=512, heads=8)
    
# Inputs to the model
v = torch.randn(1, 512, 1024)
