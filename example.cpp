#include <iostream>
#include <vector>
using namespace std;

int main() {
    int t;
    cin >> t;
    
    while (t--) {
        int n;
        cin >> n;
        
        string s(n);
        for (char& c : s) cin >> c;
        
        // 找到子序列 A 中 "1" 的数量
        int count_ones = 0;
        for (char c : s) if (c == '1') count_ones++;
        
        // 根据子序列 A 的大小调整子序列 B
        vector<char> b(count_ones);
        for (int i = 0; i < count_ones; ++i) b[i] = '1';
        
        // 将子序列 B 转换为二进制表示并输出
        string binary_result = "";
        for (char c : b) binary_result += c;
        cout << binary_result << endl;
    }
    
    return 0;
}