
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x2 = torch.nn.functional.dropout2d(x1.clone(), p=0.5, inplace=True)
        x3 = torch.randn_like(x2, requires_grad=True)
        return x3
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
