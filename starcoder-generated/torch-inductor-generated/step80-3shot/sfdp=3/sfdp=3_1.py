
class Model(torch.nn.Module):
    def __init__(self, num_heads, input_dropout_p=0):
        super().__init__()
        dim = 8
        self.linear_qkv = torch.nn.Linear(dim, 3 * num_heads * dim)
        dropout_p = input_dropout_p
        self.attn_dropout = torch.nn.Dropout(dropout_p)
        self.register_buffer("mask", self._get_attn_mask(0, 0, 4096))

    def _get_attn_mask(self, height, width, device):
        h = torch.tril(torch.ones((height, width), device=device)).view(
            1, 1, height, width)
        return h

    def forward(self, data):
        return self.forward_step(data)

    def forward_step(self, x1):
        qkv = self.linear_qkv(x1)
        q, k, v = qkv.chunk(3, dim=-1)
        