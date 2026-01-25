#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
    int T;
    cin >> T;
    
    while (T--) {
        int n;
        string s;
        cin >> n >> s;
        
        vector<int> dp(n + 1, 0);
        
        for (int i = 1; i <= n; ++i) {
            if (s[i - 1] == '0') {
                dp[i] = dp[i - 1];
            } else {
                dp[i] = dp[i - 1] + (1 << (n - i));
            }
        }
        
        cout << dp[n] << endl;
    }
    
    return 0;
}