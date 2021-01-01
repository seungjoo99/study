# -*- coding: utf-8 -*-

from MySeq import MySeq
from itertools import combinations
import copy

class MotifFinding:
    """ Class for motif finding. """

    def __init__(self, size=8, seqs=None):
        self.motif_size = size  # 찾아야하는 motif길이, default=8 (k)
        if (seqs is not None):
            self.seqs = seqs
            self.alphabet = seqs[0].alphabet()  # MySeq에 존재하는 alphabet method
        else:
            self.seqs = []
            self.alphabet = "ACGT"  # default: DNA

    def __len__(self):  # sequence의 행의 개수 (t)
        return len(self.seqs)

    def __getitem__(self, n):
        return self.seqs[n]

    def seq_size(self, i):
        return len(self.seqs[i])

    def read_file(self, fic, t):  # 파일에서 한 줄씩 MySeq에 넣어주기
        for s in open(fic, "r"):
            self.seqs.append(MySeq(s.strip().upper(), t))  # t는 type
        self.alphabet = self.seqs[0].alphabet()
        print(self.alphabet)  # DNA이면 ACGT 출력

    def create_motif_from_indexes(self, indexes):
        # pseqs = []
        res = [[0] * self.motif_size for i in range(len(self.alphabet))]
        # DNA라면 len(self.alphabet)=4 0을 motif_size만큼 갖는 리스트가 4개인 리스트
        for i, ind in enumerate(indexes):
            # i는 index, ind는 값
            subseq = self.seqs[i][ind:(ind + self.motif_size)]
            # 한 줄당 각 시작점에서 motif_size만큼 자르기 == motif 만들기
            for i in range(self.motif_size):
                for k in range(len(self.alphabet)):
                    if subseq[i] == self.alphabet[k]:
                        res[k][i] = res[k][i] + 1
            # DNA일 때 만약 subseq 안에있는 문자가 ACGT안에 존재한다면 motif당 A C G T순서로 카운트
            # subseq가 ATAGAGCT(motif 후보, 길이 8) 이면
            # res는 ([1,0,1,0,1,0,0,0],[0,0,0,0,0,0,1,0],[0,0,0,1,0,1,0,0],[0,1,0,0,0,0,0,1])
            # indexes의 길이만큼(행의 개수) 반복하므로 profile을 만든 것이다.
        return res

    def score(self, s):
        score = 0
        mat = self.create_motif_from_indexes(s)  # s로 만든 profile
        for j in range(len(mat[0])):
            maxcol = mat[0][j]
            for i in range(1, len(mat)):  # 열 하나 잡아두고 밑으로 내려가면서 가장 큰 값 찾기
                if mat[i][j] > maxcol:
                    maxcol = mat[i][j]
            score += maxcol  # 각 열마다 최댓값을 더한 것 == 각 열마다 가장 많은 염기의 개수 더한것
        return score

    from itertools import combinations
    import copy

    def greedymotif_search(self):
        n=len(self.seqs)
        l=[]
        for i in range(n):
            l.append(i)
        comb=list(combinations(l,2))
        scores = []
        ress = []
        seqs_copy = copy.deepcopy(self.seqs)

        for i in range(len(comb)):
            a,b=comb[i]
            s1 = self.seqs.pop(a)
            s2 = self.seqs.pop(b - 1)
            self.seqs.insert(0, s2)
            self.seqs.insert(0, s1)
            res = [0] * len(self.seqs)
            s = [0] * len(self.seqs)
            best_score = 0

            for i in range(0, self.seq_size(0) - self.motif_size + 1):
                for j in range(0, self.seq_size(1) - self.motif_size + 1):
                    s[0] = i
                    s[1] = j
                    sc = self.score(s)
                    if (sc > best_score):
                        best_score = sc
                        res[0] = i
                        res[1] = j
            s[0] = res[0]
            s[1] = res[1]

            for i in range(2, len(self.seqs)):
                best_score = 0
                for j in range(0, self.seq_size(i) - self.motif_size + 1):
                    s[i] = j
                    sc = self.score(s)
                    if (sc > best_score):
                        best_score = sc
                        res[i] = j
                s[i] = res[i]
            ress.append(res)
            scores.append(best_score)
            self.seqs = copy.deepcopy(seqs_copy)
        return ress , scores


import time
def test2():
    seq1 = MySeq("ATAGATCTGACTCTAGAAATGAGGCA","DNA")
    seq2 = MySeq("ACGTAGATGAACACTACGCCATCGAT","DNA")
    seq3 = MySeq("AATATAAGCGACGCAGTAATTCCCCT","DNA")
    seq4 = MySeq("AATGCAGTGACCGCATGTGCATGCCT", "DNA")
    seq5 = MySeq("TGACACATGACCGCCGCATACGGCCT", "DNA")
    mf = MotifFinding(4, [seq1,seq2,seq3,seq4,seq5])
    print ("Greedy")
    start = time.time()
    sols, scores = mf.greedymotif_search()
    fin=time.time()
    print(fin - start)
    print ("Solution: " , sols )
    print('Score:', scores)


test2()