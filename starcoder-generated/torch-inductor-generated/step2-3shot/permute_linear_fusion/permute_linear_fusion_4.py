
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
    def forward(self, x1):
        return torch.nn.functional.fold(x1.view(-1, x1.size()[-3], x1.size()[-1]), x_size=x1.size()[-2:], kernel_size=self.linear.weight.shape[0], stride=self.linear.weight.shape[0], dilation=self.linear.weight.shape[0])

#Inputs to the model
x1 = torch.randn(1, 1, 4, 4)

model = Model()
model.eval()
traced_model = torch.jit.trace(model, x1)
input_names = ['input']
output_names = [ "out" ]
torch.jit.save(traced_model, "fold.jit.pt")

