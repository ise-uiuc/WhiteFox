
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.1)
 
    def forward(self, x1, x2, x3):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.div(10)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        return dropout_qk.matmul(x3)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 4)
x2 = torch.randn(4, 4, 512)
x3 = torch.randn(512, 8)
