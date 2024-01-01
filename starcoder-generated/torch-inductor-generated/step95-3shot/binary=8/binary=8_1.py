
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.seq1 = torch.nn.Sequential(torch.nn.Conv1d(2, 2, 2, stride=2,
                                                         padding=1), torch.nn.Conv1d(2, 3, 3, stride=3, padding=1))
        self.seq2 = torch.nn.Sequential(torch.nn.Conv1d(2, 3, 2, stride=2,
                                                         padding=1), torch.nn.Conv1d(3, 3, 3, stride=3, padding=1))
    def forward(self, x1, x2):
        v1 = self.seq1(x1)
        v2 = self.seq2(x2)
        v3 = v1 + v2
        return v3
# Inputs to the model
x1 = torch.randn(2, 2, 12)
x2 = torch.randn(2, 2, 12)
