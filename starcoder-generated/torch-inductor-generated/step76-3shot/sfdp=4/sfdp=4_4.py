
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, A7, C1, F5, G6):
        A7 = A7
        C1 = C1
        F5 = F5
        G6 = G6
        B7 = A7.transpose(-2, -1)
        E1 = torch.bmm(C1, B7)
        I7 = F5.transpose(-2, -1)
        J1 = torch.bmm(G6, I7)
        E1 = torch.bmm(E1, J1)
        attn_weight = torch.softmax(E1, 0)
        return attn_weight
# Inputs to the model
A = torch.randn(4, 224, 832)
C = torch.randn(224, 832, 832)
F = torch.randn(8, 832, 832)
G = torch.randn(8, 832, 832)
