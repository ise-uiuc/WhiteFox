
torch.nn.Module() class M1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(24, 24)
 
    def forward(self, q1):
        k2 = self.linear(q1)
        v3 = self.linear(q1)
        v4 = self.linear(q1)
        qk1 = torch.matmul(k2, v3.transpose(-2, -1))
        qk2 = qk1.div(0.3)
        softmax_qk = qk2.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.78)
        output = dropout_qk.matmul(v4)
        return output

m1 = M1()
q1 = torch.randn(14, 5, 24)
__output1__ = m1(q1)

# Model
torch.nn.Module() class M2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
 
    def forward(self, q2):
        k3 = self.linear(q2)
        v5 = self.linear(q2)
        v6 = self.linear(q2)
        qk3 = torch.matmul(k3, v5.transpose(-2, -1))
        qk4 = qk3.div(0.8)
        softmax_qk = qk4.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.29)
        output = dropout_qk.matmul(v6)
        return output

m2 = M2()
q2 = torch.randn(7, 23, 2)
__output2__ = m2(q2)

# Model
torch.nn.Module() class M99(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(200, 200)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v3 = self.linear(x1)
        v5 = self.linear(x1)
        v7 = self.linear(x1)
        v9 = self.linear(x1)
        v11 = self.linear(x1)
        v13 = self.linear(x1)
        v15 = self.linear(x1)
        v17 = self.linear(x1)
        v19 = self.linear(x1)
        v21 = self.linear(x1)
        v23 = self.linear(x1)
        output = v23 + v21 + v19 + v17 + v15 + v13 + v11 + v9 + v7 + v5 + v3 + v1
        return output

m_big = M99()
v119 = torch.randn(1, 200)
__output3__ = m_big(v119)

