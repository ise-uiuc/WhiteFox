
class Model(torch.nn.Module):
    def forward(self, x):
        v1 = torch.cat((x, x), dim=1)
        lstm1 = torch.nn.LSTM(x.shape[1], x.shape[1])
        lstm2 = torch.nn.LSTM(v1.shape[1], v1.shape[1])
        v1 = lstm1(v1)[0][:, -1, :]
        v2 = lstm2(v1.unsqueeze(1))[0][:, -1, :]
        return v2.squeeze(dim=-1)
# Inputs to the model
x = torch.randn(2, 3, 2)
