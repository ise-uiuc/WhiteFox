with open(__file__.replace("utils/skip.py", "config/skip_torch.txt")) as f:
    skip_torch = f.read().split("\n")

with open(__file__.replace("utils/skip.py", "config/skip_tf.txt")) as f:
    skip_tf = f.read().split("\n")


def need_skip_torch(api_name):
    if api_name in skip_torch:
        return True
    else:
        return False


def need_skip_tf(api_name):
    if api_name in skip_tf:
        return True
    skip_keywords = ["tf.keras.applications", "Input", "get_file"]
    for keyword in skip_keywords:
        if keyword in api_name:
            return True
    return False
