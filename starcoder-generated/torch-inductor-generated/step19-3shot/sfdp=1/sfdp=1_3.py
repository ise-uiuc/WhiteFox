
class Model1(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        q1 = x1.matmul(x2.t())
        v1 = x3.matmul(x4.t())
        d1 = 32
        b1 = 64
        a1 = 1
        s1 = q1.shape
        s2 = v1.shape
        s3 = s1[:-1] + a1 * (d1,)
        s4 = s2[:-1] + b1 * (d1,)
        q2 = q1.reshape(s3).transpose(2, 3)
        v2 = v1.reshape(s4).transpose(2, 3)
        v3 = 128
        s5 = s3[:-1] + (v3,)
        s6 = s4[:-2] + (v3,)
        q3 = q2.reshape(s5)
        v4 = v2.reshape(s6)
        s7 = (b1,)
        r1 = torch.randn(s7).tril()
        r2 = r1.repeat(s1[0], 1, 1)
        r3 = r2.reshape(s3).transpose(2, 3)
        b2 = (r3 + q3).softmax(dim=-1)
        s8 = s1[:-1] + (b1,)
        s9 = s3[:-1] + (b1,)
        m1 = b2.reshape(s8)
        o1 = m1.matmul(v4)
        s10 = s3[:-1] + s7
        s11 = s5[:-1] + s7
        a2 = q1.shape
        a3 = q2.shape
        q4 = q1.reshape(a2).transpose(1, 2)
        q5 = q2.reshape(a3).transpose(1, 2)
        q6 = q4 + q5 * (r3 + r1).tril(-1).transpose(-2, -1)
        q7 = q6.transpose(1, 2)
        s12 = q7.shape
        q8 = q7.reshape(s12[:-1]+s7)
        r4 = r1.repeat(s12[0], 1, 1)
        q9 = q8 + r4
        a4 = q9.shape
        q10 = q9.reshape(a4[:-1]+(a4[-1], 1))
        s13 = torch.randn((b1, 128, b1)).softmax(dim=0)
        p1 = (s13 * q10).sum(dim=0)
        s14 = torch.randn((v3, 128)).softmax(dim=-1)
        p2 = (s14 * p1).sum(dim=-1)
        return p2

# Initializing the model
m1 = Model1()

# Inputs to the model
x3 = torch.randn(128, 32, 20, 20)
x4 = torch.randn(128, 32, 20, 20)
x1 = torch.randn(64, 128, 32, 20)
x2 = torch.randn(64, 128, 32, 32)
