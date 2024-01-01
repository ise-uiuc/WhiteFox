def load_data(file_name, multiline=False):
    with open(file_name) as f:
        if multiline:
            temp = f.read().split("\n")
            lines = []
            for line in temp:
                if line.strip() != "":
                    lines.append(line)
            return lines
        else:
            return f.read()
