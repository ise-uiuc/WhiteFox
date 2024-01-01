
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1, bias=False)
 
    def forward(self, x1):
        o1 = self.linear(x1)
        o2 = o1 * torch.clamp(o1 + 3, min=0, max=6)
        return o2 / 6

# Initializing the model
model = Model()

# Input to the model
x1 = torch.randn(1, 2).requires_grad_()

# output from the model
with torch.no_grad():
    y = model(x1)

y.backward()

# Checking the numeric output between PyTorch and the model
dy  = torch.ones_like(x1, device=x1.device)
print('PyTorch y:', y)
print('PyTorch x.grad:', x1.grad)
y_num = (x1.data * dy).sum()
print('model_num:', y_num)
y_den = dy.sum()
print('model_den:', y_den)
y_grad = y_num / y_den
print('model_grad:', y_grad)
print('model_grad.grad:', x1.grad)

with torch.no_grad():
    x1.grad.zero_()
