#include <iostream>
#include <vector>
using namespace std;

const int N = 500005;
int n;
string str;
int left[2 * N + 1], right[2 * N + 1];

int main() {
    cin >> n;
    
    for (int i = 1; i <= n; ++i) {
        char c;
        cin >> c;
        if (c == '1') {
            right[i] = right[i - 1] + 1;
            left[n + i] = left[n + i - 1] - 1;
        } else {
            left[i] = left[i - 1];
            right[n + i] = right[n + i - 1] + 1;
        }
    }
    
    long long ans = LLONG_MAX;
    for (int i = 1; i <= n; ++i) {
        long long sum = left[i] - right[i]; // 当前位置左侧 1 的数量减去右侧 1 的数量
        if (sum < ans) {
            ans = sum;
        }
    }
    
    cout << ans << endl;
    
    return 0;
}