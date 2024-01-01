
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        lstm1 = torch.nn.LSTM(2, 2)
        lstm2 = torch.nn.LSTM(2, 2)
        v1 = lstm1(torch.randn(1, 3, 2))[0]
        t = v1.new_zeros(16)
        for i in range(2):
            for j in range(3):
                t = v1[i][j][0] * torch.nn.functional.linear(v1[i][j][1], torch.randn(2, 2), torch.randn(2))
        for i in range(1):
            t = t[list(range(list(range(torch.size(t))[-1]), 0, -1))].reshape(1, 12)
        v2 = lstm2(t.view(1, 1, 2))[0]
        v3 = v2.permute(0, 2, 1)
        v4 = v2.permute(0, 3, 2, 1)
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    v3[i][k][j].add_(v4[i][k][j].view(2, 1))
        v5 = v3.flatten(start_dim=1)
        return v5
# Inputs to the model
