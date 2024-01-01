
class Model(torch.nn.Module):
    def __init__(self, d_qk, dropout_p):
        super().__init__()
        self.d_qk = d_qk
        self.proj_qk = torch.nn.Conv2d(3, d_qk, 1, stride=1, padding=1)
 
    def forward(self, x1, x2, dropout_p):
        qk = torch.matmul(self.proj_qk(x1), x2.transpose(-2, -1))
        scale_factor = self.d_qk**-0.5
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(x2)
        return output

# Initializing the model
d_qk = 32 # Specify the value of d_qk
dropout_p = 0.2 # Specify the dropout proportion
m1 = Model(d_qk, dropout_p)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
