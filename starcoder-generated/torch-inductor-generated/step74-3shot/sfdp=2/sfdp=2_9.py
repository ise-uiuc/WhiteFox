
class Model(torch.nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
    
    def forward(self, v1, v2):
        qk = torch.matmul(v1, v2.transpose(-2, -1))
        scaled_qk = qk.div(1 / self.dropout_p)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v1)
        return output

# Initializing weights
p = 0.4
m = Model(p)

# Inputs to the model
x1 = torch.randn(1, 4, 10, 8)
x2 = torch.randn(1, 4, 8, 10)
