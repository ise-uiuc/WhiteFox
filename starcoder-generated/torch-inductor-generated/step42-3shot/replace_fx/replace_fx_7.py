
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input_tensor):
        t1 = torch.nn.functional.dropout(input_tensor, p=0.1)
        t2 = torch.nn.functional.dropout(input_tensor, p=0.1)
        return (t1, t2)
# Inputs to the model
input_tensor = torch.randn(1, 2, 2)
