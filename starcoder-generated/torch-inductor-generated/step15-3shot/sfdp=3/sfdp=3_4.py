
class Model(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(d_in, d_out) / d_in**0.5)
        self.key = torch.nn.Parameter(torch.randn(d_in, d_out) / d_in**0.5)
        self.value = torch.nn.Parameter(torch.randn(d_in, d_out) / d_in**0.5)
        self.scale_factor = d_in**0.5
        self.dropout_p = 0.1
 
    def forward(self, x1):
        qk = torch.matmul(x1, self.key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
m = Model(d_in=64, d_out=64)

# Inputs to the model
x1 = torch.randn(1, 64, 64)
