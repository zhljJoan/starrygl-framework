        
#include <bits/stdc++.h>
using namespace std;
string s;
int n;
int main(){
	cin>>s;
	cin>>n;
	int len=s.size();
	int g=s.size();
	for(int i=0;i<n;i++){
		char op;
		cin>>op;
		if(op=='L'&&g){
			g--;
		}
		if(op=='D'&&g<len){
			g++;
		}
		if(op=='B'&&g){
			len--;
			g--;
			s.erase(g,1);
		}
		if(op=='P'){
			string c;
			cin>>c;
			len++;
			s.insert(g,c);
            g++;
		}
        cout<<g<<" "<<s<<endl;
	}
	cout<<s<<endl;
	return 0;
}
/*
dmih
11
B
B
P x
L
B
B
B
P y
D
D
P z


 */