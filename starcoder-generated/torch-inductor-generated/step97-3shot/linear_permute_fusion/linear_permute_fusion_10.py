
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(10, 20, 2)
    def forward(self, x0):
        h0 = torch.rand(2, 2, 20)
        c0 = torch.rand(2, 2, 20)
        v1, (h1, c1) = self.lstm(x0, (h0, c0))
        v2 = v1.permute(1, 0, 2)
        return v2
# Inputs to the model
tensor_input = torch.randn(5, 3, 10)
