
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 1)
    def forward(self, x1):
        x2 = torch.nn.functional.softmax(self.linear1(x1).permute(0, 2, 1), dim=1)
        x3 = torch.nn.functional.softmax(self.linear2(x1).permute(0, 2, 1), dim=1)
        return torch.nn.functional.cosine_similarity(x2, x3)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
