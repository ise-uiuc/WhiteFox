
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        s1 = qk.flatten(2).transpose(-2, -1)
        s2 = s1.div(4)
        s3 = torch.nn.functional.softmax(s2, dim=-1)
        d1 = torch.nn.functional.dropout(s3, p=0.005)
        o1 = d1.transpose(-2, -1).matmul(x1.flatten(2))
        o2 = o1.transpose(-2, -1).contiguous().view(1, 1, 2, 2)
        return o2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 4)
x2 = torch.randn(1, 2, 4)
