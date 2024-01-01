
class Model(torch.nn.Module):
    def __init__(self, dim_model, num_heads):
        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.scale = dim_model ** -0.5
        self.qkv_proj = torch.nn.Conv2d(dim_model, dim_model * 3, 1, stride=1, padding=0)
 
    def forward(self, qv):
        batch_size, dim_feature, h, w = qv.shape
        qkv = self.qkv_proj(qv)
        qkv = qkv.reshape(batch_size, 3, -1, h * w)
        qkv = qkv.transpose(1, 2).reshape(batch_size, -1, 3, h * w)
        query, key, value = qkv.chunk(3, dim=-2)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk * self.scale
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        output = output.reshape(batch_size, -1, h * w).transpose(1, 2).reshape(batch_size, -1, dim_feature, h, w)
        return output
 
# Initializing the model
m = Model(dim_model, num_heads)

# Inputs to the model
qv1 = torch.randn(1, 32, 64, 64)
