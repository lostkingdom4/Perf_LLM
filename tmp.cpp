
#include <set>
std::set<int> s;
void add(int x) {
    s.insert(x);
}
bool find(int x) {
    return s.find(x) != s.end();
}
void remove(int x) {
    s.erase(x);
}
bool empty() {
	return s.empty();
}
int getMin() {
	return *s.begin();
}
