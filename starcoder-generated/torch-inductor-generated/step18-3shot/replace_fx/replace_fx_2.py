
class M1(nn.Module):
    def __init__(self):
        super(M1, self).__init__()

    # torch.nn.functional.dropout
    def forward(self, x):
        x = F.dropout(x, p=0.5, training=self.training)
        x = x * 3
        return x
# Inputs to the model
input_tensor = torch.randn((5,5))
