
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = torch.nn.Linear(32, 4)
        self.wk = torch.nn.Linear(32, 4)
        self.wv = torch.nn.Linear(32, 8)
        self.inv_scale_factor = 2 ** 4
        self.dropout_p = 0.2
 
    def forward(self, x1, x2):
        qk = torch.matmul(self.wq(x1), self.wk(x2).transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(self.wv(x2))
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
x2 = torch.randn(2, 32)
