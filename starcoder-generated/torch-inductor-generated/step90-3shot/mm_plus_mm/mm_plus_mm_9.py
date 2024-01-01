
class Model(nn.Module):
    def forward(self, q1, q2, q3, q4):
        # Matrix multiplication
        t1 = torch.mm(q1, q2)
        t2 = torch.mm(q3, q4)
        # Addition
        t3 = t1 + t2
        # Matrix multiplication
        t4 = torch.mm(t3, q2)
        return t4 + t1
# Inputs to the model
q1 = torch.randn(3, 2)
q2 = torch.randn(2, 3)
q3 = torch.randn(3, 2)
q4 = torch.randn(2, 3)
