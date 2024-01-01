
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = torch.nn.Dropout(0.1)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.dropout3 = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.matmul = torch.nn.MatMul()
 
    def forward(self, x1, x2):
        p1 = x1.matmul(x2.transpose(-2, -1))
        p2 = p1 / 4
        p3 = self.softmax(p2)
        p4 = self.dropout1(p3)
        p5 = p4.mm(x1)
        p6 = self.matmul(self.dropout2(p4), self.dropout3(self.softmax(x2/4)))
        return p6, p6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(20, 32, 24)
x2 = torch.randn(20, 24, 24)

__output1__, __output2__ = m(x1, x2)

