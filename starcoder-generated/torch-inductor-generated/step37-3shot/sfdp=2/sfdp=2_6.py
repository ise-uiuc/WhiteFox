
class Model(torch.nn.Module):
    def __init__(self):
        pass
 
    def forward(self, x1, x2):
        a1 = torch.matmul(x1, x2.transpose(-2, -1))
        a2 = a1.div(1)
        a3 = torch.nn.functional.softmax(a2, dim=-1)
        a4 = torch.nn.functional.dropout(a3, p=0.2)
        