
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
 
    def forward(self, x1, x2, x3, x4):
        q = self.linear(x1)
        k = self.linear(x2)
        v = self.linear(x3)
        scale_factor = self.linear(x4)
        inv_scale_factor = scale_factor.softmax(dim=0).unsqueeze(-1)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.4)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
model = Model()

# Inputs to the model
x1 = torch.randn(4, 8)
x2 = torch.randn(4, 8)
x3 = torch.randn(4, 8)
x4 = torch.randn(4, 8)
