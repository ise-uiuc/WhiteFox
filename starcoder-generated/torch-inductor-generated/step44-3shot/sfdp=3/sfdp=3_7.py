
class Model(torch.nn.Module):
    def forward(self, t0, t1, t2, t3):
        t4 = torch.matmul(t0, torch.transpose(t1, -2, -1))
        t5 = t4 * t3
        t6 = torch.nn.functional.softmax(t5, dim=-1)
        t7 = torch.nn.functional.dropout(t6, p=0.35)
        t8 = torch.matmul(t7, t2)
        return t8

# Initializing the model
m = Model()
# Inputs to the model
t0 = torch.randn(8, 10, 1, 64, 64)
t1 = torch.randn(8, 10, 1, 64, 64)
t2 = torch.randn(8, 10, 3, 64, 64)
t3 = 0.15
