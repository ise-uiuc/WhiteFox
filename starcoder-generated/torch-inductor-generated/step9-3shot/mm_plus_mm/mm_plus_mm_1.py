
class Model(torch.nn.Module):
    def forward(self, w, x1, x2, x3, x4, x5, x6):
        v1 = torch.mm(w, w)
        t1 = torch.mm(w, w)
        t2 = torch.mm(w, w)
        t3 = torch.mm(w, w)
        t4 = torch.mm(w, w)
        t5 = torch.mm(w, w)
        t6 = torch.mm(w, w)
        t7 = torch.mm(w, w)
        t8 = torch.mm(w, w)
        v3 = torch.mm(w, w)
        return v1 * v3
# Inputs to the model
w = torch.randn(5, 5, requires_grad=True)
x1 = w + 1
x2 = w + 2
x3 = w + 3
x4 = w + 4
x5 = w + 5
x6 = w + 6
