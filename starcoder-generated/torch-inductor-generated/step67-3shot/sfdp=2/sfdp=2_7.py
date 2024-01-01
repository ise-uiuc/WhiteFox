
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = torch.nn.MatMul()
 
    def forward(self, x1, x2, dropout_p=0.5):
        qk = self.matmul(x1, x2)
        qk = qk.div(np.sqrt(256))
        softmax_qk = torch.nn.functional.softmax(qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = self.matmul(dropout_qk, x3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1000000, 256)
x2 = torch.randn(1, 1000000, 256)
x3 = torch.randn(1, 100000, 256)
dropout_p = 0.5
