
class Model(torch.nn.Module):
    def __init__(self, num_heads, num_queries, num_keys, mlp_dim, dropout_p=0.1, scale_factor=1.0 / math.sqrt(8)):
        super().__init__()
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.num_keys = num_keys
        self.mlp_dim = mlp_dim
        self.dropout_p = dropout_p
        self.scale_factor = scale_factor
 
        self.key = torch.nn.Linear(self.num_keys, self.num_heads, bias=True)
        self.query = torch.nn.Linear(self.num_queries, self.num_heads, bias=True)
        self.value = torch.nn.Linear(self.num_keys, self.num_heads, bias=True)
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.num_heads, self.mlp_dim, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(self.dropout_p),
            torch.nn.Linear(self.mlp_dim, self.num_heads, bias=True),
            torch.nn.Dropout(self.dropout_p)
        )
 
    def forward(self, inputs):
        key = self.key(inputs).permute(0, 2, 3, 1)
        query = self.query(inputs).permute(0, 2, 1, 3).contiguous()
        value = self.value(inputs).permute(0, 2, 1, 3).contiguous()
 
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
 
        out = output.permute(0, 2, 1, 3).contiguous()
        out = self.mlp(out)
 
        return out

# Initializing the model
m = Model(16, 64, 256, 128)

# Inputs to the model
x1, x2 = torch.randn(1, 256, 64), torch.randn(1, 256, 256)
