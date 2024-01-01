class ResultInv:
    def __init__(self, status, value, grad=None, err_msg="") -> None:
        self.status = status
        self.value = value
        self.grad = grad
        self.err_msg = err_msg

    def is_fail(self):
        return self.status == "fail"
