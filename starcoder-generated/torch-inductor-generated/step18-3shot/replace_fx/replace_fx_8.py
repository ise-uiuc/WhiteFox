
class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.a = torch.rand(5)
        self.b = torch.rand(5)
    def forward(self, input):
        y1 = torch.rand_like(input)
        y3 = torch.add(input, y1)
        y2 = torch.dropout(self.a)
        y4 = torch.rand_like(input)
        y = torch.add(y3, y4)
        return y
# Inputs to the model
input1 = torch.zeros(5)
