
class Model(torch.nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, q, k, v, scale_factor):
        scaled_qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = scaled_qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output
 
# Initializing the model
m = Model(dropout_p=0.5)

# Inputs to the model
q = torch.randn(1, 64, 56)
k = torch.randn(1, 64, 56)
v = torch.randn(1, 64, 56)
scale_factor = torch.tensor([10.0])
