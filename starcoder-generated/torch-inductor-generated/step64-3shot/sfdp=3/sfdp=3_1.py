
class Model(torch.nn.Module):
    def __init__(self, qdim, kdim, vdim):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.3)
 
    def forward(self, query, key, value, scale_factor=1.0):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        return torch.matmul(dropout_qk, value)


# Initializing the model
qdim, kdim, vdim = 8, 12, 20
m = Model(qdim, kdim, vdim)

# Inputs to the model
query = torch.randn(1, qdim, 20)
key = torch.randn(1, kdim, 20)
value = torch.randn(1, vdim, 20)
