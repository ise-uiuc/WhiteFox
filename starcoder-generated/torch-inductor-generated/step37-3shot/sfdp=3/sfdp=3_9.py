
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = Query()
        self.key = Key()
        self.value = Value()
 
    def forward(self, x1):
        q = self.query(x1)
        k = self.key(x1)
        v = self.value(x1)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 256, 256)
