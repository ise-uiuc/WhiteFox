
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(v1, x3)
        x1 = x1 + v2
        # Insert the correct operation in the comment (i.e. return v1, return v2, or return v2 + inp)
        return v2, x1
# Inputs to the model
