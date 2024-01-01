
class Model(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qk = torch.nn.Linear(dim, dim)
        self.value = torch.nn.Linear(dim, dim)
 
    def forward(self, x1, x2, x3, input_mask):
        qk = self.qk(x1)
        value = self.value(x2)
        inv_scale_factor = torch.rsqrt(torch.sum(qk.mul(qk), dim=-1, keepdim=True))
        qk = qk.mul(inv_scale_factor).softmax(dim=-1)
        input_mask = input_mask.to(torch.float)
        dropout_qk = torch.nn.functional.dropout(qk, p=0.5)
        qk = qk.mul(input_mask)
        output = qk.matmul(value)
        output = output.unsqueeze(0)

        return output

# Initializing the model
m = Model(8)

# Inputs to the model
x1 = torch.randn(1, 8, 10, 16)
x2 = torch.randn(1, 8, 16, 10)
x3 = torch.randn(1, 10, 16)
input_mask = torch.randn(1, 8, 10, 16)
