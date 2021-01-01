from Bio import SeqIO
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML

# Load the file and check that the protein contains 1248 aminoacids.

record = SeqIO.read('uniprot-yourlist_M20201112A94466D2655679D1FD8953E075198DA8107208D.fasta', format='fasta') #load file
#Uniprot site에서 basket에 담아두고 다운받으니까 파일이름이 위와같아서 따로 파일이름을 바꾸진 않음
print(record.seq) #sequence 확인
print(len(record.seq)) #길이 1248임을 확인하기

# Using BLASTP, search for sequences with high similarity to this sequence, in the“swissprot” database

result_handle=NCBIWWW.qblast('blastp','swissprot',record.format('fasta')) #blastp, swissprot database
save_file = open('interl-blast.xml', 'w') #inerl-blast.xml
save_file.write(result_handle.read())
save_file.close()
result_handle.close()
#qblast를 한번 수행하는데 시간이 오래 걸려 xml파일로 저장해두고 불러오는 식

# Check which the global parameters were used in the search: the database, the substitution matrix, and the gap penalties.
result_handle = open('interl-blast.xml')
blast_record = NCBIXML.read(result_handle)
print('<parameters>')
print('database:' + blast_record.database) #사용한 database
print('matrix:' + blast_record.matrix) # 사용한 substitution matrix
print('Gap penalties:', blast_record.gap_penalties) #사용한 gap_penalties

#List the best alignments returned, showing the accession numbers of the sequences, the E value of the alignments, and the alignment length


result=[] #각 alignments에서 best local alignment 정보 저장

for i in range(10):
    alignment=blast_record.alignments[i]
    accession=alignment.accession #데이터 수탁번호
    e_value=alignment.hsps[0].expect #best local alignment의 e_value
    length=alignment.hsps[0].align_length #best local alignment의 length
    result.append('accession:'+accession+' e_value:'+str(e_value)+' length:'+str(length))

for i in result:
    print(i)

# Repeat the search restricting the target sequences to the organism S. cerevisiae
result_handle2 = NCBIWWW.qblast('blastp', 'swissprot', record.format('fasta'), entrez_query='S.cerevisiae')

#entrez_query에 Saccharomyces cerevisiae[organism] 설정 추가
save_file2 = open('interl-blast2.xml', 'w')
save_file2.write(result_handle2.read())
save_file2.close()
result_handle2.close()


result_handle2 = open('interl-blast2.xml')
blast_record2 = NCBIXML.read(result_handle2)

for alignment in blast_record2.alignments: #7개 정도라서 top 10처럼 몇 개만 뽑지 않고 다 출력
    print("Accession: " + alignment.accession)
    print("E-value:", alignment.hsps[0].expect)
    print("Length: ", alignment.hsps[0].align_length)
    print ("Query start: ", alignment.hsps[0].query_start) #query의 시작점
    print ("Sbjct start: ", alignment.hsps[0].sbjct_start) #subject의 시작점
    print (alignment.hsps[0].query[0:90]) #각각 서열을 90개 정도 출력
    print (alignment.hsps[0].match[0:90]) #어느 부분이 같은지 보여줌
    print (alignment.hsps[0].sbjct[0:90])
    print ("")
