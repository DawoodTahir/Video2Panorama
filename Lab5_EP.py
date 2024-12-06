# with open('text.txt','w') as outfile:
#     outfile.write('Im the first One here\n')
#     outfile.write('Im the Second One here')


# with open('text.txt','r') as outfile:
#     x=outfile.readline()
#     print(x)



##AM counter

from collections import Counter
def AM_Counter(file):
    dict={}
    with open(file,'r') as outfile:
        x=outfile.readlines()
        y=''.join(x.strip() for x in x).split()
        q=Counter(y)
                
print(AM_Counter('text.txt'))