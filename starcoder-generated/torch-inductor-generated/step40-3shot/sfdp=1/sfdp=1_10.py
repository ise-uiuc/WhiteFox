
class Model(torch.nn.Module):
    def __init__(self, output_dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.dim_head = output_dim / self.n_heads
        self.qk_v_proj = torch.nn.Linear(output_dim * 2, output_dim * 3, bias=False)
        self.ff_proj = torch.nn.Linear(output_dim, output_dim)
 
    def forward(self, inputs, valid_length=None):
        kqv = torch.cat([self.qk_v_proj(inputs).chunk(3, dim=-1)], dim=0)
        k, q, v = kqv[0], kqv[1], kkv[2]
        scale_factor = 1 / self.dim_head ** 0.5
        qkp = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
        softmax_qkp = torch.nn.functional.softmax(qkp, dim=-1)
        dropout_qkp = torch.nn.functional.dropout(softmax_qkp, p=dropout_p)
        output = torch.matmul(dropout_qkp, v)
        return self.ff_proj(output)
 
# Initializing the model
m = Model(output_dim=80)
 
# Inputs to the model
inputs = torch.randn(4, 80, 80)
