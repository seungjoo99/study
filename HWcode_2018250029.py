import copy
score_matrix=[]
def score(match, mismatch,gap_penalty,seq1,seq2):
    global score_matrix #다른 함수에서 사용해야하므로 global 변수로 만들어주기
    words=[None, '_'] #score_matrix에서 indel 나타내기 위한 '_'이 필요
    n=len(seq1)
    m=len(seq2)
    for i in range(n):
        words.append(seq1[i]) #seq1에 있는 문자들 추가
    for j in range(m):
        words.append(seq2[j]) #seq2에 있는 문자들 추가
    wordset=[]
    for k in range(len(words)):
        for v in words:
            if v not in wordset:
                wordset.append(v) #중복제거하여 문자 종류만 가질 수 있도록 wordset만들기
    l=len(wordset)
    for i in range (0,l):
        score_matrix.append([0]*l) # lXl 크기의 행렬만들기
    score_matrix[0]=wordset #첫 번째 행은 none, _, 문자 종류들
    for i in range(1,l):
        score_matrix[i][0]=wordset[i] #첫 번째 열도 none, _, 문자 종류들이 되도록
    for i in range(2,l):
        score_matrix[i][i]=match #두 문자 같을 때 match
        score_matrix[1][i]=gap_penalty #indel 나타내는 gap_penalty
        score_matrix[i][1]=gap_penalty
    i=2
    while i<l-1:
        for j in range(i+1,l):
            score_matrix[i][j]=mismatch #나머지는 mismatch로 채우기
            score_matrix[j][i]=mismatch
        i+=1
    return score_matrix #score_matrix 반환


B = []
S = []

def global_alignment_with_tie(seq1, seq2, score_matrix):
    global B #print_align_with_ties에서 사용하므로 global 변수
    global S
    rows = len(seq1)
    columns = len(seq2)

    S = [[0 for i in range(rows + 1)] for j in range(columns + 1)]
    B = [[0 for i in range(rows + 1)] for j in range(columns + 1)]
    S[0][0] = 0
    B[0][0] = ['시작']

    for n in range(1, rows + 1):
        for m in range(1, columns + 1):
            num1 = seq1[n - 1]
            score1 = score_matrix[0].index(num1) #해당 문자가 score_matrix에서 index확인
            S[n][0] = S[n - 1][0] + score_matrix[score1][1] # +gap_penalty
            B[n][0] = ['up'] #seq2에서 insert

            num2 = seq2[m - 1]
            score2 = score_matrix[0].index(num2)
            S[0][m] = S[0][m - 1] + score_matrix[1][score2]  #+gap_penalty
            B[0][m] = ['left'] #seq2에서 deletion

            if n != 0 and m != 0:
                a = S[n - 1][m] + score_matrix[score1][1]
                b = S[n][m - 1] + score_matrix[1][score2]
                c = S[n - 1][m - 1] + score_matrix[score1][score2]
                #score1=score2라면 +match, 같지않다면 +mismatch
                S[n][m] = max(a, b, c) #score가 가장 커질 때의 점수

                altn = [] #S[n][m]이 여러개면 모든 path를 고려해주어야 하므로
                if S[n][m] == a:
                    altn.append('up')
                if S[n][m] == b:
                    altn.append('left')
                if S[n][m] == c:
                    altn.append('diag')
                B[n][m] = altn
    score = S[rows][columns] #S에서 가장 오른쪽 밑이 도착점이므로
    return score #optimal score 반환


def print_align_with_ties(B, seq1, seq2):
    global C #경로가 여러개가 아닌 곳은 다시 복원시켜주어야 하므로 B에 변화 생기기전에 B저장해둔 변수
    n = len(seq1)
    m = len(seq2)
    l = len(B[n][m]) - 1  # 최대가 되는 방향이 1개면 0, 여러개라면 1,2
    global l1
    global l2
    check = True
    global ans #함수 자체를 재귀해서 사용하므로 종료조건에서만 update 되도록 global 변수로 만듦
    while check:

        if n == 0 and m == 0:  # 시작점 도착 = 종료조건
            l3 = ''.join(l1) #[‘A’, ‘T’, ‘_’]처럼 문자 append되어있는 것 하나의 문자열로
            l4 = ''.join(l2)
            ans.append([l3, l4]) #path list에 추가
            break

        if B[n][m][l] == 'diag':  # 마지막 도착점에서 시작
            l1.insert(0, seq1[n - 1])  #해당 문자열의 마지막 원소 추가
            l2.insert(0, seq2[m - 1])
            sub1 = seq1[n - 1] #다시 seq 되돌릴 때 필요하므로 저장
            sub2 = seq2[m - 1]
            seq1 = seq1[:n - 1] #seq 업데이트
            seq2 = seq2[:m - 1]
            del B[n][m][l] # 확인한 곳 제거

            print_align_with_ties(B, seq1, seq2)
            # 다시 seq 되돌리기
            seq1 = seq1 + sub1
            seq2 = seq2 + sub2
            l = len(B[n][m]) - 1
            l1 = l1[1:] #추가했던 원소 하나씩 되돌리면서 다른 path 확인할 수 있도록
            l2 = l2[1:]
            if l < 0: #원래 경로가 하나밖에 없었던 곳
                B[n][m] = copy.deepcopy(C[n][m]) #맨 처음 B와 같은 C에서 불러와서 되돌려주기. C는 변하면 안되므로 deepcopy
                check = False
                break

        if B[n][m][l] == 'left':
            l1.insert(0, '_')
            l2.insert(0, seq2[m - 1])
            sub2 = seq2[m - 1]
            seq2 = seq2[:m - 1]
            del B[n][m][l]
            print_align_with_ties(B, seq1, seq2)

            seq2 = seq2 + sub2 #seq2만 줄어들었으니까 seq2만 되돌려주면 됨
            l1 = l1[1:]
            l2 = l2[1:]
            l = len(B[n][m]) - 1
            if l < 0:
                B[n][m] = copy.deepcopy(C[n][m])
                check = False
                break

        if B[n][m][l] == 'up':
            l1.insert(0, seq1[n - 1])
            l2.insert(0, "_")
            sub1 = seq1[n - 1]
            seq1 = seq1[:n - 1]
            del B[n][m][l]
            print_align_with_ties(B, seq1, seq2)

            seq1 = seq1 + sub1 #seq1만 줄어들었으니까 seq1만 되돌려주면 됨
            l = len(B[n][m]) - 1
            l1 = l1[1:]
            l2 = l2[1:]
            if l < 0:
                B[n][m] = copy.deepcopy(C[n][m])
                check = False
                break

#test1
'''
seq1='TAGAAT'
seq2='TAAGAT'
match=1
mismatch=0
gap_penalty=0
score(match,mismatch,gap_penalty,seq1,seq2)
print(score_matrix)
print(global_alignment_with_tie(seq1,seq2,score_matrix))
print(B)
print(S)

C = copy.deepcopy(B)
l1 = []
l2 = []
ans = []
print_align_with_ties(B,seq1,seq2)
print(ans)
'''
#test2

seq1='NPKLINAL'
seq2='NPQLIHAL'
match=2
mismatch=-2
gap_penalty=-1
score(match,mismatch,gap_penalty,seq1,seq2)
print(score_matrix)
print(global_alignment_with_tie(seq1,seq2,score_matrix))
print(B)
print(S)

C = copy.deepcopy(B)
l1 = []
l2 = []
ans = []
print_align_with_ties(B,seq1,seq2)
print(ans)
