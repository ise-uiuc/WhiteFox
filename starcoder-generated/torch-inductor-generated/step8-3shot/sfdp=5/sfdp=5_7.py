
class Model(torch.nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.emb = torch.nn.Embedding(196, 720)
    self.enc = torch.nn.Sequential(
      torch.nn.Conv1d(720, 576, 2, 1),
      torch.nn.Hardsigmoid(),
      torch.nn.Conv1d(576, 512, 2, 1, dilation=2),
      torch.nn.Hardsigmoid(),
      torch.nn.Conv1d(512, 512, 2, 1, dilation=4),
      torch.nn.Hardsigmoid(),
      torch.nn.Conv1d(512, 512, 2, 1, dilation=8),
      torch.nn.Hardsigmoid(),
      torch.nn.Conv1d(512, 256, 2, 1),
      torch.nn.Hardsigmoid()
    )
    self.query = torch.nn.Linear(512, 768)
    self.key = torch.nn.Linear(512, 768)
    self.value = torch.nn.Linear(512, 768)
    self.att_weight = torch.nn.Conv1d(1, 252, 1)
 
  def forward(self, x1, x2):
    x55 = torch.reshape(self.emb(x1), [1, -1, 24, 28])
    x56 = torch.reshape(self.emb(x2), [1, -1, 24, 28])
    x59 = self.enc(x55)
    v1 = self.query(x59)
    v2 = self.key(x56)
    v3 = v1 @ v2.transpose(-2, -1) / math.sqrt(v1.size(-1))
    x60 = self.att_weight(v3.unsqueeze(0)).squeeze(0)
    v4 = torch.nn.functional.softmax(x60, dim=1).unsqueeze(-1)
    v5 = v4 @ v3.unsqueeze(1).transpose(-2, -1).squeeze(1)
    v6 = self.value(v5)
    return v6

# Inputs to the model
x1 = torch.randint(8, [1, 28], dtype="long").unsqueeze(0)
x2 = torch.randint(8, [1, 70], dtype="long").unsqueeze(0)
