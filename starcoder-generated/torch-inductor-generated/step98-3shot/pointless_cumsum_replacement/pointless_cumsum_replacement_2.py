
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = torch.full([100, 100], 1, dtype=torch.int32, layout=torch.strided, device=torch.device('cuda:0'), pin_memory=False)
        t2 = convert_element_type(t1, torch.float16)
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x = 1
