int func(int x, int y) {
    int result = x OPERATOR y;  // result is related to x and y
    return result;
}

int main() {
  int a = 0x12345678;
  int b = 0x1234;
  short truncated_result = (short)(func(a,b));
  return truncated_result;
}