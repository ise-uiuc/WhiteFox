
class Model(torch.nn.Module):
    def forward(self, A, B, C, D):
        a = torch.sigmoid(A+C)
        b = torch.sigmoid(B+D)
        c = torch.tanh(A+B+C+D)
        d = torch.tanh(A+D)
        return a*b+a*c+b*c+c*d+d*a
