
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w_qs = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.w_ks = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.v = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2):
        q1 = self.w_qs(x1)
        k1 = self.w_ks(x2)
        v1 = self.v(x2)
        qk = torch.matmul(q1, k1.transpose(-2, -1))
        scaled_qk = qk.div(10.0)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = dropout_qk.matmul(v1)
        return output
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
