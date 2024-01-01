
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Conv2d(3, 50, 3, stride=3, padding=1)
        self.key = torch.nn.Conv2d(3, 50, 3, stride=3, padding=1)
        self.value = torch.nn.Conv2d(3, 50, 3, stride=3, padding=1)
        self.inv_scale_factor = 100000
        self.dropout_p = 0
        
    def forward(self, x1):
        q = self.query(x1)
        k = self.key(x1)
        v = self.value(x1)
        qkey = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qkey.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
