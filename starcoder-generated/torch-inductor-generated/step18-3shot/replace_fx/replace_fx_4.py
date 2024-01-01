
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    def forward(self, a):
        # a.shape = [8, 2, 3]
        a = a.unsqueeze(dim=-1)
        a = torch.nn.functional.dropout(input=a, p=0.5, training=self.training, inplace=False)
        # a.shape = [8, 2, 1, 3]
        a = a.squeeze(dim=-1)
        return a
# Inputs to the model
a1 = torch.rand([8, 2, 3])
