
class Model(torch.nn.Module):
    def __init__(self, scaling_factor, dropout):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dropout = dropout

    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.mul(self.scaling_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout)
        output = dropout_qk.matmul(x2)
        return output

# Initializing the model
m = Model(0.6, 0.8)

# Inputs to the model
x1 = torch.randn(1, 32, 4, 4)
x2 = torch.randn(1, 32, 10, 10)
