
class Model(torch.nn.Module):
    def __init__(self, scale_factor, dropout_p):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p

    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = torch.nn.MultiheadAttention(33, 8, 2) # 8 heads for each query and key
scale_factor = 256 ** -0.5 # a scaling factor of 256^-0.5 was chosen to normalize the dot product
dropout_p = 0.2 
m = Model(scale_factor=scale_factor, dropout_p=dropout_p)

# Inputs to the model
q = torch.randn(1, 64, 33)
k = torch.randn(1, 33, 33)
v = torch.randn(1, 33, 64)
