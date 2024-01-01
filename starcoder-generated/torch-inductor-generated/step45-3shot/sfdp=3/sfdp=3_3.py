
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.nn.Linear(4, 8)
        self.m2 = torch.nn.Linear(8, 8)
   
    def forward(self, x1):
        qk = torch.matmul(x1, self.m1.weight.t())
        v1 = self.m1(x1)
        v2 = torch.matmul(v1, self.m2.weight.t())
        scale_factor = -0.5
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=1)
        dropout_p = 0.2  # Please adjust this value on your own if there are dropout layers.
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 4)
__input__ = x1
