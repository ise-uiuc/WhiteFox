
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.2):
        super().__init__()
        self.proj_dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, q, k, v, scale_factor):
        qkv_combined = torch.matmul(q, k.transpose(-2, -1))
        rescaled_qkv = qkv_combined.div(scale_factor)
        softmax_qkv = rescaled_qkv.softmax(dim='heads')
        dropout_qkv = self.proj_dropout(softmax_qkv)
        output = dropout_qkv.matmul(v)
        return output

# Initializing the model
m = Model()

# Initializing query, key and values
q = torch.rand(1, 24, 768).reshape(1, 24, 6, 12)
k = torch.rand(1, 24, 768).reshape(1, 24, 12, 6)
v = torch.rand(1, 24, 768).reshape(1, 24, 12, 6)
scale_factor = 24
