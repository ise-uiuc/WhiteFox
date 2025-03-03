// test_code.c
int add_function(int x, int y) {
  int result = x + y;
  return result;
}

int main() {
  int a = 10;
  int b = 20;
  int sum = add_function(a, b);
  return sum;
}