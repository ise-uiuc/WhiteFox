
class Model(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, nhead: int, window_size: int, dropout: float, use_bias: bool = True):
        super().__init__()
        self.in_proj_weight = torch.nn.Parameter(torch.empty(nhead, in_dim, window_size))
        if use_bias:
            self.in_proj_bias = torch.nn.Parameter(torch.empty(nhead, in_dim * window_size))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj_weight = torch.nn.Parameter(torch.empty(nhead, in_dim, window_size))
        if use_bias:
            self.out_proj_bias = torch.nn.Parameter(torch.empty(nhead, out_dim))
        else:
            self.register_parameter("out_proj_bias", None)
        self.reset_parameters()
 
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_proj_weight.size(1))
        self.in_proj_weight.data.uniform_(-stdv, stdv)
        if self.in_proj_bias is not None:
            self.in_proj_bias.data.zero_()
        self.out_proj_weight.data.uniform_(-stdv, stdv)
        if self.out_proj_bias is not None:
            self.out_proj_bias.data.zero_()
 
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask: torch.Tensor, need_weights: bool = True):
        r