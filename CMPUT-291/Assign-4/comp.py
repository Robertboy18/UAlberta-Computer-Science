z = int(input())
lines =[] 
for r in open("input.txt","r+"):
    lines = r.split(",")
k = []
for i in range(len(lines)):
    if lines[i] != 'x':
        k.append(lines[i])
k = [int(i) for i in k]
t = [i + z - z%i for i in k ]
l = min(t) - z
for i in range(len(t)):
    if t[i] == min(t):
        g = i
final = l*k[g]
print(k,t,final) 