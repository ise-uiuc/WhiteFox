
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        t1 = torch.matmul(x1, x2.transpose(-2, -1))
        t2 = t1 * 1.7071067811865476
        t3 = torch.nn.functional.softmax(t2, dim=-1)
        t4 = torch.nn.functional.dropout(t3, p=0.10000000149011612)
        t5 = torch.matmul(t3, x3)
        t6 = torch.matmul(t4, x4)
        concat = torch.cat([t5, t6], dim=0)
        return concat

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
x2 = torch.randn(1, 4)
x3 = torch.randn(1, 8)
x4 = torch.randn(1, 16)
