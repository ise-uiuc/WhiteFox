
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale_factor = math.sqrt(self.dim)
        self.w_q = torch.nn.Linear(dim, dim * num_heads, bias=False)
        self.w_k = torch.nn.Linear(dim, dim * num_heads, bias=False)
        self.w_v = torch.nn.Linear(dim, dim * num_heads, bias=False)
 
    def forward(self, x1, x2):
        q, k, v = self.w_q(x1), self.w_k(x2), self.w_v(x1)
        q = self._reshape(q, x2.size(0))
        k = self._reshape(k, x1.size(0))
        v = self._reshape(v, x1.size(0))
 
        scale_factor = self.scale_factor
        inv_scale_factor = 1 / scale_factor
        dropout_p = 0.1
        return torch.nn.functional.dropout(torch.softmax(torch.matmul(q, k.transpose(-2, -1)).div(inv_scale_factor), dim=-1), p=dropout_p).matmul(v)
 
    def _reshape(self, x, batch_size):
        dim = x.size(-1)
        splitted = torch.split(x, split_size_or_sections=self.num_heads, dim=-1)
        return torch.cat([tensor.view(batch_size, dim, 1, 1) for tensor in splitted], dim=2).view(-1, dim)

# Initializing the model
m = Model(dim=256)

# Inputs to the model
x1 = torch.randn(100, 256)
x2 = torch.randn(100, 256)
