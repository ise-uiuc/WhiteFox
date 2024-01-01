
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = torch.matmul
        self.div = torch.div
        self.softmax = torch.nn.Softmax(dim=-1)
        self.nn_functional_dropout = torch.nn.functional.dropout
        self.matmul1 = torch.matmul
        self.mul = torch.mul
        self.add = torch.add
    
    def forward(self, x1, x2, x3, scale_factor, dropout_p):
        qk = self.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = self.div(qk, scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.nn_functional_dropout(softmax_qk, p=dropout_p)
        output = self.matmul1(dropout_qk, x3)
        return self.mul(x1, output) + self.add(x1, output)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
scale_factor = 1
dropout_p =.5
