
embeddings_file="data/out4015.emb"
output="data/test_sentences.emb"
output2="data/test.emb"
output3="data/train.emb"


with open(output2,'w') as out:
    with open(embeddings_file) as f:
        for i in range(25000):
            x=f.readline()
#            if i in range(48000,49000) :#or i in range(22500,25000):
            if i%4==1 :
               out.writelines(x)

with open(output3,'w') as out:
    with open(embeddings_file) as f:
        for i in range(49000):
            x=f.readline()
#            if i in range(48000,49000) :#or i in range(22500,25000):
            if i%4!=1 :
               out.writelines(x)



