
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_conv = torch.nn.Conv2d(3, 64, 1, stride=1, padding=1)
        self.k_conv = torch.nn.Conv2d(64, 64, 1, stride=1, padding=1)
        self.v_conv = torch.nn.Conv2d(64, 64, 1, stride=1, padding=1)
 
    def forward(self, x1):
        q = self.q_conv(x1)
        k = self.k_conv(x1)
        v = self.v_conv(x1)
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv = 1 / qk.size(-1)
        qk = qk * inv
        softmax_qk = qk.softmax(dim=-1)
        drop_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = torch.matmul(drop_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
