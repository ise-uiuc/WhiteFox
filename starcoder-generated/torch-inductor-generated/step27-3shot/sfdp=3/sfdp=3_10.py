
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, dim_out):
        super().__init__()
        self.num_heads = num_heads
        self.wq = torch.nn.Linear(dim, dim, bias=False)
        self.wk = torch.nn.Linear(dim, dim, bias=False)
        self.wv = torch.nn.Linear(dim, dim, bias=False)
        self.wo = torch.nn.Linear(dim, dim_out, bias=False)
        self.scale_factor = math.sqrt(dim_out / self.num_heads)
        self.dropout_p = 0.2
 
    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        query = self.wq(x1)
        key = self.wk(x2)
        value = self.wv(x2)
        flat_query = query.view((batch_size * self.num_heads, -1, query.shape[-1]))
        flat_key = key.view((batch_size * self.num_heads, -1, key.shape[-1]))
        flat_value = value.view((batch_size * self.num_heads, -1, value.shape[-1]))
        flat_scaled_qk = torch.matmul(flat_query, flat_key.transpose(1, 2)) * self.scale_factor
        flat_softmax_qk = flat_scaled_qk.softmax(dim=-1)
        flat_dropout_qk = torch.nn.functional.dropout(flat_softmax_qk, p=self.dropout_p)
        output = flat_dropout_qk.matmul(flat_value)
        output = output.view((batch_size, -1, output.shape[-1])).transpose(1, 2)
        return self.wo(output)

# Initializing the model
m = Model(dim=20, num_heads=8, dim_out=128)

# Inputs to the model
x1 = torch.randn(1, 5, 20)
