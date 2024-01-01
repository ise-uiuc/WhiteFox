
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.2
    def forward(self, input):
        t1 = torch.rand_like(input)
        t1 += 0.0033333333
        t2 = torch.zeros_like(t1)
        t2 += 0.11535001535000937
        m = torch.rand_like(t2) < self.dropout_p
        t2 = torch.where(m, t1, t2)
        output = torch.clip(t2, max=1.0)
        return output
# Inputs to the model
input = torch.rand_like(torch.ones(10, 10))
