
A = torch.randn(4, 4)
B = torch.randn(4, 4)
C = torch.randn(4, 4)
D = torch.randn(4, 4)
E = torch.randn(4, 4)
F = torch.randn(4, 4)
model = torch.nn.Sequential(torch.nn.Linear(4,4), torch.nn.ReLU(), torch.nn.Linear(4, 4))
for i in range(5):
    A = torch.mm(A, C) + B
    B = A
    C = B + E
    model.eval()
    model.add_module(str(i), torch.nn.Sequential(torch.nn.Linear(4,4), torch.nn.ReLU()))
    model.train()
D = A + F
model(A, B, C, D, E, F)
# Inputs to the model
A = torch.randn(4, 4)
B = torch.randn(4, 4)
C = torch.randn(4, 4)
D = torch.randn(4, 4)
E = torch.randn(4, 4)
F = torch.randn(4, 4)
