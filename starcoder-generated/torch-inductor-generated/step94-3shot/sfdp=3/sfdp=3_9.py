
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_q = torch.nn.Linear(20, 5)
        self.linear_k = torch.nn.Linear(40, 10)
        self.linear_v = torch.nn.Linear(30, 5)
 
    def forward(self, x1):
        x2 = self.linear_q(x1)
        x3 = self.linear_k(x1)
        x4 = x2.view(x2.shape[0], -1, 1, )
        x5 = x3.view(x3.shape[0], 1, -1)
        qk = torch.matmul(x4, x5)
        scale_factor = (x2.shape[0] * x2.shape[-1]) ** -0.5
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_p = 0.1
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        value = self.linear_v(x1)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20)
