# -*- coding: utf-8 -*-

from MySeq import MySeq


class MotifFinding:
    """ Class for motif finding. """
    
    def __init__(self, size = 8, seqs = None):
        self.motif_size = size #찾아야하는 motif길이, default=8
        if (seqs is not None):
            self.seqs = seqs
            self.alphabet = seqs[0].alphabet() #MySeq에 존재하는 alphabet method
        else:
            self.seqs = []
            self.alphabet = "ACGT" # default: DNA
        
    def __len__ (self): #sequence의 행의 개수
        return len(self.seqs)
    
    def __getitem__(self, n):
        return self.seqs[n]
    
    def seq_size (self, i):
        return len(self.seqs[i])
    
    def read_file(self, fic, t): #파일에서 한 줄씩 MySeq에 넣어주기
        for s in open(fic, "r"):
            self.seqs.append(MySeq(s.strip().upper(),t)) #t는 type
        self.alphabet = self.seqs[0].alphabet()
        print(self.alphabet) #DNA이면 ACGT 출력
        
    def create_motif_from_indexes(self, indexes):
        #pseqs = []
        res = [[0]*self.motif_size for i in range(len(self.alphabet))]
        #DNA라면 len(self.alphabet)=4 0을 motif_size만큼 갖는 리스트가 4개인 리스트
        for i,ind in enumerate(indexes):
            #i는 index, ind는 값
            subseq = self.seqs[i][ind:(ind+self.motif_size)]
            #한 줄당 각 시작점에서 motif_size만큼 자르기 == motif 만들기
            for i in range(self.motif_size):
                for k in range(len(self.alphabet)):
                    if subseq[i] == self.alphabet[k]:
                        res[k][i] = res[k][i] + 1
            #DNA일 때 만약 subseq 안에있는 문자가 ACGT안에 존재한다면 motif당 A C G T순서로 카운트
            #subseq가 ATAGAGCT(motif 후보, 길이 8) 이면
            #res는 ([1,0,1,0,1,0,0,0],[0,0,0,0,0,0,1,0],[0,0,0,1,0,1,0,0],[0,1,0,0,0,0,0,1])
            #indexes의 길이만큼(행의 개수) 반복하므로 profile을 만든 것이다.
        return res    

    def score(self, s):
        score = 0
        mat = self.create_motif_from_indexes(s) #s로 만든 profile
        for j in range(len(mat[0])):
            maxcol = mat[0][j]
            for i in range(1, len(mat)): #열 하나 잡아두고 밑으로 내려가면서 가장 큰 값 찾기
                if mat[i][j] > maxcol: 
                    maxcol = mat[i][j]
            score += maxcol #각 열마다 최댓값을 더한 것 == 각 열마다 가장 많은 염기의 개수 더한것
        return score
   

    # EXHAUSTIVE SEARCH - 모든 가능한 시작점 집합에 대해 score 계산해서 best score인 시작점 찾기
    def next_solution (self, s):
        next_sol= [0]*len(s)
        pos = len(s) - 1
        # s[pos]는 주어진 s의 마지막 줄을 의미, 마지막 줄에서 시작점이 될 수 있는 곳까지 고려
        # 그 지점에 도달했으면 pos를 -1해줘서 그 윗줄로 올라간다고 생각
        while pos >=0 and s[pos] == self.seq_size(pos) - self.motif_size:
            pos -= 1
        if (pos < 0): 
            next_sol = None
        else:
            for i in range(pos): 
                next_sol[i] = s[i]
            next_sol[pos] = s[pos]+1;
            for i in range(pos+1, len(s)):
                next_sol[i] = 0
        return next_sol

    def exhaustive_search(self):
        best_score = -1
        res = []
        s = [0]* len(self.seqs)
        while (s!= None):
            sc = self.score(s)
            if (sc > best_score): #s 업데이트해가면서 best_score갖는 시작점 리스트찾기
                best_score = sc
                res = s
            s = self.next_solution(s)
        return res


    # # BRANCH AND BOUND     
    def next_vertex (self, s):
        res =  []
        if len(s) < len(self.seqs): # internal node -> down one level
            for i in range(len(s)): 
                res.append(s[i])
            res.append(0) #끝에 0 넣어줌으로써 밑의 level로 들어갈 수 있도록
        else: # bypass
            pos = len(s)-1 
            while pos >=0 and s[pos] == self.seq_size(pos) - self.motif_size:
                pos -= 1
            if pos < 0: res = None # last solution 마지막 가능한 시작점 리스트까지 도달한 것임
            else:
                for i in range(pos): res.append(s[i])
                res.append(s[pos]+1)
        return res
    
    
    def bypass (self, s):
        res = []
        pos = len(s) - 1
        while pos >= 0 and s[pos] == self.seq_size(pos) - self.motif_size:
            pos -= 1
        if pos < 0: #더 이상 bypass할 곳이 없음
            res = None
        else:
            for i in range(pos):
                res.append(s[i])
            res.append(s[pos] + 1)
        return res


    def branch_and_bound (self):
        best_score = -1
        res = []
        s = [0]
        while (s != None):
            if len(s) < len(self.seqs):
                opt_sc = self.score(s) + (len(self.seqs) - len(s)) * (self.motif_size)
                #현재 s까지의 score계산하고 나머지는 모두 s에 맞게 조절하여 score을 가장 최대로 만든 것
                if opt_sc < best_score: #만든 최대값이 현재까지의 best_score보다 작다면 child로 내려갈 필요가 없음
                    s = self.bypass(s)
                else: #child로 내려가서 확인해보기
                    s = self.next_vertex(s)

            else: #제일 밑 leaves이므로 exhaustive 방법과 같은 원리
                sc = self.score(s)
                if sc > best_score:
                    best_score = sc
                    res = s
                s = self.next_vertex(s)
        return res


def test1():
    sm = MotifFinding()
    print(sm.alphabet)
    print (len(sm.alphabet))
    sm.read_file("exampleMotifs.txt","DNA")
    sol = [25,20,2,55,59]
    print (len(sm.alphabet))
    si = sm.create_motif_from_indexes(sol)
    print (si)  
    sa = sm.score(sol)
    print(sa)


    
def test2():
    seq1 = MySeq("ATAGAGCTGA","DNA")
    seq2 = MySeq("ACGTAGATGA","DNA")
    seq3 = MySeq("AAGATAGGGG","DNA")
    mf = MotifFinding(3, [seq1,seq2,seq3])
    
    print ("Exhaustive:")
    sol = mf.exhaustive_search()
    print ("Solution: " , sol)
    print ("Score: ", mf.score(sol))
    
    print ("\nBranch and Bound:")
    sol2 = mf.branch_and_bound()
    print ("Solution: " , sol2)
    print ("Score:" , mf.score(sol2))

def test3():
    mf = MotifFinding()
    mf.read_file("exampleMotifs.txt","DNA")
    print ("Branch and Bound:")
    sol = mf.branch_and_bound()
    print ("Solution: ", sol)
    print ("Score:", mf.score(sol))

import time
def test4():
    mf=MotifFinding()
    mf.read_file('sequ.txt','DNA')
    start1 = time.time() #exhaustive_search() 전 시간
    sol1=mf.exhaustive_search()
    fin1= time.time() #exhaustive_search() 후 시간
    print(sol1,mf.score(sol1))
    print(fin1-start1) #exhaustive_search() 걸린 시간

    start2=time.time()  #branch_and_bound() 전 시간
    sol2=mf.branch_and_bound()
    fin2=time.time()  #branch_and_bound() 후 시간
    print(sol2, mf.score(sol2))
    print(fin2-start2) #branch_and_bound() 걸린 시간


test1()
print()
test2()
print()
test3()
print()
test4()
