
class Model(torch.nn.Module):
    def forward(self, w, x):
        y=torch.mm(w,x)+torch.mm(x,w)
        return y
# Inputs to the model
w = torch.randn(5, 5)
x = torch.randn(5, 5)
