
class Model(torch.nn.Module):
    def __init__(self, d_model, num_heads, scale_factor=1 / (d_model // num_heads)):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_proj = torch.nn.Linear(d_model, 3 * d_model)
        self.scale_factor = scale_factor
        self.dropout = torch.nn.Dropout(0.3)
 
    def forward(self, x1):
        (q,k,v) = torch.chunk(self.qkv_proj(x1), 3, dim=-1)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(288, 4)

# Inputs to the model
x1 = torch.randn(1, 49, 288)
