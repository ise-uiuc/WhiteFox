
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    def forward(self, args, kwargs):
        t1 = torch.nn.functional.dropout(input_1=args, p=0.3)
        t2 = torch.nn.functional.dropout(input=kwargs, p=0.4)
        return t2 + t1
# Inputs to the model
args1 = torch.zeros(3, 4)
kwargs1 = torch.ones(3, 4)
