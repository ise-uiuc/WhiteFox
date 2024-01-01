
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 8)
        self.linear1 = nn.Linear(8, 2)
        self.linear2 = nn.Linear(2, 2)
        self.linear3 = nn.Linear(1, 128)
        self.linear4 = nn.Linear(128, 1)
        self.gru = nn.GRU(input_size=1, hidden_size=4)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax()
    def forward(self, x):
        x = self.layers(x)
        x = nn.Tanh()(x)
        x = self.linear1(x)
        x = nn.Tanh()(x)
        x = self.linear2(x)
        x = nn.Tanh()(x)
        x = torch.unsqueeze(x, -1)
        x = self.linear3(x)
        x = nn.Tanh()(x)
        x = torch.transpose(x, 1, 2).contiguous()
        batch_size = x.size(0)
        h0 = torch.randn(1, batch_size, 4)
        out, _ = self.gru(x, h0)
        out = torch.sum(out, dim=1)
        out = torch.unsqueeze(out, 2)
        out = self.linear4(out)
        out = torch.squeeze(out)
        out = self.softmax(out)
        return out
# Inputs to the model
x = torch.randn(3, 4)
