
class Model(torch.nn.Module):
    def forward(self, Q, K, V, scale, dropout):
        S = torch.matmul(Q, K.transpose(-2, -1))
        y1 = S.div(scale)
        y2 = torch.nn.functional.softmax(y1, dim=-1)
        y3 = torch.nn.functional.dropout(y2, p=dropout)
        output = torch.matmul(y3, V)
        return output

# Inputs to the model
Q = torch.randn(20, 15, 512)
K = torch.randn(20, 10, 512)
V = torch.randn(20, 10, 512)
scale = 3.0
dropout = 0.8
