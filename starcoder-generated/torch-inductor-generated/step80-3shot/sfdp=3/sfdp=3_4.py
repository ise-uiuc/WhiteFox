
class Model(torch.nn.Module):
    def __init__(self, scaling_factor=0.1, dropout_p=0.5):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dropout_p = dropout_p
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk * self.scaling_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model(scaling_factor=0.1, dropout_p=0.5)

# Inputs to the model
q = torch.randn(1, 8, 1024)
k = torch.randn(1, 8, 1024)
v = torch.randn(1, 8, 1024)
