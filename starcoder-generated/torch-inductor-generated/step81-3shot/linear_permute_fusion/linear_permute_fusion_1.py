
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTMCell(3, 6).to('cuda')
    def forward(self, x1):
        v0 = self.lstm(x1[0])
        v1 = self.lstm(v0)
        return v0.permute(0, 2, 1) + v1
# Inputs to the model
x1 = (torch.randn(1, 3, 6, device='cuda'), torch.randn(1, 6, device='cuda'))
