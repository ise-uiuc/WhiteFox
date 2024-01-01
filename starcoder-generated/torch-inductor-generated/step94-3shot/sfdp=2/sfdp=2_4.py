
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(3, 8, bias=False)
        self.key = torch.nn.Linear(5, 10, bias=False)
        self.value = torch.nn.Linear(10, 15, bias=False)
 
    def forward(self, x1, x2, inv_scale_factor, dropout_p=0.1):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the models
model1 = Model()
model2 = Model()
model3 = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(2, 5)
inv_scale_factor = torch.tensor(float(1 / sqrt(3 * 5)))
__output1__ = model1(x1, x2, inv_scale_factor)
__output2__ = model2(x1, x2, inv_scale_factor, dropout_p=0.2)
__output3__ = model3(x1, x2, inv_scale_factor, dropout_p=0.0)