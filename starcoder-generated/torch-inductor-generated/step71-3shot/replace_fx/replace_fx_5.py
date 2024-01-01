
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(x, model_input):
        t0 = torch.pow(x, 3)
        t1 = torch.add(x, model_input, alpha=1)
        dropout0 = F.dropout(t0, p=0.05)
        t2 = torch.pow(dropout0, 3)
        t2 = torch.add(t2, model_input, alpha=1)
        t3 = torch.pow(t1, 3)
        t4 = torch.add(t1, model_input, alpha=1)
        dropout1 = F.dropout(t3, p=0.05)
        t5 = torch.pow(dropout1, 3)
        t5 = torch.add(t5, model_input, alpha=1)
        return t5
# Inputs to the model
x = torch.randn((10, 2, 2))
model_input = torch.randn((10, 2, 2))
