
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        v2 = torch.nn.Conv2d(6, 10, kernel_size=5, stride=1, padding=0)
        v3 = torch.nn.functional.dropout(v2(v1(x1)), p=0.5, training=True)
        v4 = torch.nn.MaxPool2d(kernel_size=2)
        v5 = torch.nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=0)
        v6 = torch.sigmoid() (v5(v4(v3)))
        v7 = torch.nn.Conv2d(20, 12, kernel_size=5, stride=1, padding=0)
        v8 = torch.nn.MaxPool2d(kernel_size=2)
        v9 = torch.nn.Conv2d(12, 12, kernel_size=5, stride=1, padding=0)
        v10 = torch.nn.Conv2d(12, 2, kernel_size=5, stride=1, padding=0)
        v11 = torch.nn.Conv2d(2, 12, kernel_size=5, stride=1, padding=0)
        v12 = torch.nn.Conv2d(12, 10, kernel_size=5, stride=1, padding=0)
        v13 = torch.nn.Conv2d(10, 22, kernel_size=5, stride=1, padding=0)
        v14 = torch.nn.Conv2d(22, 10, kernel_size=5, stride=1, padding=0)
        v15 = torch.nn.Conv2d(10, 6, kernel_size=5, stride=1, padding=0)
        v16 = torch.nn.Conv2d(6, 18, kernel_size=5, stride=1, padding=0)
        v17 = torch.nn.Conv2d(18, 34, kernel_size=5, stride=1, padding=0)
        v18 = torch.nn.Sigmoid(v17(v16(v15(v14(v13(v12(v11(v10(v9(v8(v7(v6(v3)))))))))))))
        return v18
# Inputs to the model
x1 = torch.randn(1, 28, 28)
