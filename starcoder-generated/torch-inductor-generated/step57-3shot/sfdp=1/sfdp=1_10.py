
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.dropout2 = torch.nn.Dropout(p=0.1)
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        inv_scale_factor = 1.0 / 4.0
        dropout_qk = self.dropout2(self.dropout1(qk.div(inv_scale_factor)).softmax(dim=-1))
        v1 = dropout_qk.matmul(x2)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16, 8, 64, 64)
x2 = torch.randn(23, 8, 32, 32)
