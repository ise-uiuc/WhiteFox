
class Model(torch.nn.Module):
    def forward(self, input):
        _1986 = torch.mm(input, input)
        _2 = self.my_mm(input, _1986)
        return _2
    def my_mm(self, _0, _1):
        _2 = torch.mm(_0, _1)
        return _2
# Inputs to the model
input = torch.randn(100, 100)
