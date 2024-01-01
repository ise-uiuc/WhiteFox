
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x1):
        x2 = F.dropout(x1, p=0.5)
        x3 = F.dropout(x2, p=0.5)
        x4 = F.feature_dropout(x3, p=0.5)
        x5 = F.dropout(x4, p=0.5)
        x6 = F.feature_alpha_dropout(x5, p=0.5)
        return x6
# Inputs to the model
x1 = torch.randn(1, 2, 2)
