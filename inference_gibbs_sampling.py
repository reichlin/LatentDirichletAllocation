from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer
import numpy as np
import scipy.special as scsp
import math

def print_results(K,n_words,corpus, dictionary, phi):
    print("Print processing...")
    results_indexes=np.zeros((n_words,K))
    results_words=np.zeros((n_words,K))
    array=[]
    D=len(corpus)
    for k in range(K):
        words=[]
        n=0
        while(n<n_words):
            max_val=0
            for d in range(D):
                idx=np.argmax(phi[d][:,k])
                if(phi[d][idx][k]>max_val):
                    idx_n=idx
                    idx_d=d

            phi[idx_d][idx_n][k]=0
            if(dictionary[corpus[idx_d][idx_n]] in words):
                n = n-1
            else:
                words.append(dictionary[corpus[idx_d][idx_n]])
            n+=1
            #TODO:check if word already in the array for the topic k, in case go to next one

        array.append(words)
    #matrix=np.reshape(array,(K,n_words))
    np.savetxt('matrix.txt',array,fmt='%s')
    print(array)



def E_step(corpus, K, D, alpha, beta, phi, gamma):
    print("E")
    loglikelihood=0
    for d in range(D):
        print(d)
        N = len(corpus[d])
        count=0
        convergence=0
        while convergence==0:
            new_phi = phi[d]
            new_gamma = gamma[d]
            for n in range(N):
                for k in range(K):
                    #new_phi[n][k] = np.exp(scsp.digamma(gamma[d][k]) - scsp.digamma(np.sum(gamma[d])))

                    new_phi[n][k] = beta[k][corpus[d][n]] * np.exp(scsp.digamma(gamma[d][k]) - scsp.digamma(np.sum(gamma[d])))
                    #print(gamma[d][k], scsp.digamma(gamma[d][k]), scsp.digamma(np.sum(gamma[d])))
                #print(np.sum(new_phi[n]), new_phi[n])
                new_phi[n] = new_phi[n] / np.sum(new_phi[n]) # phi normalization
            new_gamma = alpha + np.sum(new_phi, axis=0)
            #print(alpha)

            phi[d] = new_phi
            gamma[d] = new_gamma

            #########SET RIGHT CONVERGENCE############
            #if (new_gamma-gamma[d]<vi_treshold && new_phi)
            count += 1
            if count > 10:
                convergence = 1

        #TODO: LOGLIKELIHOOD=LOGLIKELIHOOD+L(gamma,phi;alpha,beta)
    return phi, gamma


def M_step(alpha, beta, phi, gamma, corpus, D, V, K):
    print("M")
    for d in range(D):
        N = len(corpus[d])
        for n in range(N):
            for k in range(K):
                beta[k][corpus[d][n]] += phi[d][n][k]
    for k in range(K):
        beta[k, :] = beta[k, :] / np.sum(beta[k, :])

    alpha = 100
    log_alpha = np.log(alpha)
    count = 0
    convergence = 0
    while convergence == 0:
        alpha = np.exp(log_alpha)
        if math.isnan(alpha):
            print("NaN dio bestia")
            alpha=100
            log_alpha = np.log(alpha)


        dL = D * (K * scsp.polygamma(1, K*alpha) - K * scsp.polygamma(1, alpha))
        for d in range(D):
            for k in range(K):
                dL += np.sum(scsp.digamma(gamma[d][k]))
            dL -= K * np.sum(scsp.digamma(np.sum(gamma[d][k])))
        d2L = D * (K**2 * scsp.polygamma(2, K*alpha) - K * scsp.polygamma(2, alpha))

        log_alpha -= dL / (d2L * alpha + dL)

        count += 1
        if count > 10 or np.abs(dL)<1e-5:
            convergence = 1

    #Todo: check loglikelihood convergence

    return alpha, beta

    #alpha = np.ones(K) * np.exp(log_alpha)
    '''
    while convergence == 0:
        alpha_sum = np.sum(alpha)

        z = - scsp.polygamma(1, alpha_sum)
        h = D * scsp.polygamma(1, alpha) + z
        g = D * (scsp.digamma(alpha_sum) - scsp.digamma(alpha))
        for d in range(D):
            for k in range(K):
                g[k] += scsp.digamma(gamma[d][k])
            g -= scsp.digamma(np.sum(gamma[d]))
        c = np.sum(g/h) / (1/z + np.sum(1/h))

        alpha -= (g - c) / h

        #########SET RIGHT CONVERGENCE############
        # if (new_gamma-gamma[d]<vi_treshold && new_phi)
        count += 1
        if count > 3:
            convergence = 1
    '''



tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

###LOADING RAW FILE AND STORE THEM IN THE DOCUMENT GROUP
'''doc_set=[]
for i,file_name in enumerate(gutenberg.fileids()):
    doc_set=np.append(doc_set,gutenberg.raw(file_name))
    '''
file = open("/Users/gigi/Desktop/KTH/AdvancedML/project/ap.txt", "r")

line = file.readline()
doc_set = []
count=0
while line:
    line = file.readline()
    line = file.readline()
    line = file.readline()
    doc_set.append(file.readline())
    line = file.readline()
    line = file.readline()
    #count += 1
    #if count == 10:
     #   break

file.close()

texts = []
for i in doc_set:
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    # add tokens to list
    texts.append(stopped_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

#print(dictionary.doc2idx(texts[0],unknown_word_index=-1))
corpus=[]
for text in texts:
    corpus.append(dictionary.doc2idx(text, unknown_word_index=-1))

K = 100
V = len(dictionary)
D = len(corpus)
alpha = 0.5
beta = np.random.rand(K, V)
for k in range(K):
    beta[k, :] = beta[k, :] / np.sum(beta[k, :])

phi = []
gamma=[]
for d in range(D):
    N = len(corpus[d])  # number of words in the document
    phi.append(np.ones((N, K)) * 1 / K)
    gamma.append(np.ones(K) * (alpha + N / K))

#EM loop
count = 0
convergence = 0
while convergence == 0:
    phi, gamma = E_step(corpus, K, D, alpha, beta, phi, gamma)
    alpha, beta = M_step(alpha, beta, phi, gamma, corpus, D, V, K)

    #todo:#########SET RIGHT CONVERGENCE############
    # if (new_gamma-gamma[d]<vi_treshold && new_phi)
    count += 1
    print(count)
    if count > 10:
        convergence = 1



print_results(K,20,corpus,dictionary,phi)
print("bella regaz")




