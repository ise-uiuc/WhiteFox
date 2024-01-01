
model = torch.nn.Sequential(torch.nn.Conv1d(1, 1, 2, bias=False))
# Inputs to the model
x1 = torch.zeros(1, 1, 2)
