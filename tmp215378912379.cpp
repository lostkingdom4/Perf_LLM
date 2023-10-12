
#include <bits/stdc++.h>
template <class T>
concept Integral = std::is_integral_v<T>;
int main() {
  auto f = []<Integral T>(T a, T b) { return a * b; };
  int a, b;
  std::cin >> a >> b;
  std::cout << f(a, b) << std::endl;
}
