
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.dropout_p = dropout_p
        self.scale_factor = torch.sqrt(torch.Tensor([29.0644])))
    
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 32, 48, 32)
k = torch.randn(1, 12, 48, 64)
v = torch.randn(1, 24, 48, 64)
