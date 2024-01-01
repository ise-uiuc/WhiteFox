
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(128, 64, bias=False)
        self.key = torch.nn.Linear(128, 64, bias=False)
        self.value = torch.nn.Linear(128, 128, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(0.5)
 
    def forward(self, x2):
        q = self.query(x2)
        k = self.key(x2)
        v = self.value(x2)
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = torch.Tensor([10])
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing model
m = Model()

# Inputs to the model
x2 = torch.randn(128, 128)
