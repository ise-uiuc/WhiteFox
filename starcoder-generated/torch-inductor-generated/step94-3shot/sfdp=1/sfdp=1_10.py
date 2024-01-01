
class Model(torch.nn.Module):
    def __init__(self, n_heads, d_value, scale_factor, dropout_p):
        super().__init__()
        self.n_heads = n_heads
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p
        self.attention = DotProductAttention(scale_factor=scale_factor, dropout_p=dropout_p)
        
    def forward(self, input_tensor):
        x1 = torch.randn(1, 32, 24, 512)
        q = x1.transpose(1, 2).reshape(1, 24, 32, 512 // self.n_heads)
        