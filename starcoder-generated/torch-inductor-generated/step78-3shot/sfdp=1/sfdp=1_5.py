
class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels=1000, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.in_proj_weight = torch.nn.Parameter(torch.Tensor(in_channels, in_channels))
        self.out_channels = out_channels
        self.head_dim = out_channels // num_heads
        self.scaling = self.head_dim ** -0.5
        self.in_proj_bias = torch.nn.Parameter(torch.empty(3 * in_channels))
        self.out_proj_weight = torch.nn.Parameter(torch.Tensor(out_channels, out_channels))
        self.out_proj_bias = torch.nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()
 
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.in_proj_weight)
        self.in_proj_bias.data.fill_(0.)
        torch.nn.init.xavier_uniform_(self.out_proj_weight)
        self.out_proj_bias.data.fill_(0.)
 
    def forward(self, query, value, key_padding_mask):
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        w = torch.cat([self.in_proj_weight.unsqueeze(0)] * self.num_heads, dim=0)
        w = w.view(self.num_heads, -1, self.in_channels)
        a = torch.baddbmm(self.in_proj_bias, query, w, beta=1.0, alpha=self.scaling)
        b = a
        q3 = b.transpose(0, 1)
        w = torch.cat([self.out_proj_weight.unsqueeze(0)] * self.num_heads, dim=0)
        w = w.view(self.num_heads, -1, self.out_channels)
        
        b = b.view(batch, self.head_dim, len)
        b = torch.bmm(a, w)
        b = b.view(batch, len, self.out_channels)
        b = b.view(batch, self.num_heads, len, -1)
        b = b.transpose(0, 1)
        return b

# Initializing the model
m = Model(in_channels=512)

# Inputs to the model
query = torch.randn(27, 512, 64)
value = torch.randn(27, 512, 64)
key_padding_mask = torch.zeros(27, 64)
