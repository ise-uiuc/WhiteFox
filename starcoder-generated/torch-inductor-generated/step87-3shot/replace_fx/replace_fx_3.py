
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        y1 = torch.nn.functional.dropout(x1, p=0.5)
        y2 = torch.nn.functional.dropout(y1, p=0.5)
        return y2
    def dropout(self, input):
        return F.dropout(input, p=0.4)
# Inputs to the model
x1 = torch.zeros([1, 3, 3], requires_grad=True)
