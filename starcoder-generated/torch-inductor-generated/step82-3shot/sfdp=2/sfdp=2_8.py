
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(4, 2)
        self.key = torch.nn.Linear(4, 2)
        self.value = torch.nn.Linear(4, 2)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(0.2)
 
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
 
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        x = dropout_qk.matmul(v)
 
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
