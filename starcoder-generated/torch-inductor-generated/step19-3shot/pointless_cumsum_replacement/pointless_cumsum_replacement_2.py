
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.modules.rnn.GRUCell(10, 20, bias=False)
    def forward(self, input, hx):
        h1, c = self.model(input, hx)
        return h1, torch.cat([h1, c], 1)
# Inputs to the model (input, hx)
input = torch.randn(1, 10, device='cuda:0')
hx = torch.randn(1, 20, device='cuda:0')
