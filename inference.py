from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer
import numpy as np
import scipy.special as scsp
import math
import time


def E_step(corpus, K, D, alpha, beta, phi, gamma):
   for d in range(D):
       oldgamma = gamma[d]
       dif = 10
       count = 0
       while count < 50 and dif > 1e-6:

           N = len(corpus[d])

           exp = np.ndarray(K)
           for k in range(K):
               exp[k] = math.exp(digamma(gamma[d][k]))

           for n in range(N):
               sum_phi = 0
               for k in range(K):
                   phi[d][n][k] = beta[k][corpus[d][n]] * exp[k]
                   sum_phi += phi[d][n][k]
               phi[d][n] = phi[d][n] / sum_phi # phi normalization
           gamma[d] = alpha + np.sum(phi[d], axis=0)
           count += 1
           difgamma = gamma[d] - oldgamma
           dif = np.sum(difgamma) / len(gamma[d])
   return [phi, gamma]


def M_step(alpha, beta, phi, gamma, corpus, D, V, K):
   beta = np.zeros((K,V))
   for k in range(K):
       for v in range(V):
           for tuple in word_map[v]:
               beta[k][v] += phi[tuple[0]][tuple[1]][k]
       beta[k, :] = beta[k, :] / np.sum(beta[k, :])

   alpha = 100
   log_alpha = np.log(alpha)
   count = 0
   convergence = 0

   while convergence == 0:
       alpha = np.exp(log_alpha)
       if math.isnan(alpha):
           alpha = 50/K * 10;
           print("NaN", alpha)
           log_alpha = np.log(alpha);

       dL = D * (K * scsp.polygamma(1, K*alpha) - K * scsp.polygamma(1, alpha))
       for d in range(D):
           for k in range(K):
               dL += scsp.digamma(gamma[d][k])
           dL -= K * scsp.digamma(np.sum(gamma[d][k]))
       d2L = D * (K**2 * scsp.polygamma(2, K*alpha) - K * scsp.polygamma(2, alpha))
       log_alpha -= dL / (d2L * alpha + dL)
       count += 1
       if (count > 100):
           print("Newton-Raphson - ", "dl: ", dL, "count: ", count)
           convergence = 1
   alpha = np.ones(K) * np.exp(log_alpha)
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
   return [alpha, beta]


#pass phi, gamma, gamma_sum, corpus for a document d only!!
def likelihood(phi, gamma, gamma_sum, alpha, beta, corpus, K, N):
   L = np.real(scsp.loggamma(K*alpha)) - K * np.real(scsp.loggamma(alpha))
   l = 0
   for k in range(K):
       l += scsp.digamma(gamma[k])
   L += l * (K*alpha - 1)
   L += K*scsp.digamma(gamma_sum)*(1-alpha)
   for n in range(N):
       l = 0
       for k in range(K):
           l += scsp.digamma(gamma[k]) - scsp.digamma(gamma_sum)
           l += np.log(beta[k][corpus[n]])
           l += np.log(phi[n][k])
           L += l * phi[n][k]
   L -= np.real(scsp.loggamma(gamma_sum))
   for k in range(K):
       L += np.real(scsp.loggamma(gamma[k]))
       L -= (gamma[k] - 1) * (scsp.digamma(gamma[k]) - scsp.digamma(gamma_sum))
   return L


def print_results_beta(beta, dictionary, K, V, nwords):
   for k in range(K):
       best = np.argpartition(beta[k], -nwords)[-nwords:]
       for i in range(nwords):
           print(dictionary[best[i]])
       print("--------------------")

def print_results(phi, corpus, dictionary, K, D, nwords):
    for k in range(K):
        storage = np.zeros((2, nwords))
        for d in range(D):
            N = len(corpus[d])
            for n in range(N):
                min = np.inf
                imin = 0
                for i in range(nwords):
                    if storage[0, i] < min:
                        min = storage[0, i]
                        imin = i

                if phi[d][n][k] > min:
                    newword = 0
                    for i in range(nwords):
                        if corpus[d][n] == storage[1, i]:
                            newword = 1
                            break
                    if newword == 0:
                        storage[0, imin] = phi[d][n][k]
                        storage[1, imin] = corpus[d][n]

        for i in range(nwords):
            print(dictionary[storage[1, i].astype(int)], storage[0, i])
        print("--------------------")


def digamma(x):
   x = x + 6
   p = 1 / (x ** 2)
   p = (((0.004166666666667 * p - 0.003968253986254) * p +
         0.008333333333333) * p - 0.083333333333333) * p
   p = p + np.log(x) - 0.5 / x - 1 / (x - 1) - 1 / (x - 2) - 1 / (x - 3) - 1 / (x - 4) - 1 / (x - 5) - 1 / (x - 6)
   return p


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
file = open("/home/luca/Desktop/Uni/Advanced Machine Learning/project/dataset/ap/ap.txt", "r")
D = 300
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
   count += 1
   if count > D:
       break
file.close()

texts = []
for i in doc_set:
   # clean and tokenize document string
   raw = i.lower()
   tokens = tokenizer.tokenize(raw)
   # remove stop words from tokens
   stopped_tokens = [i for i in tokens if not i in en_stop]
   # stem tokens
   #stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

   final_tokens = [i for i in stopped_tokens if len(i) > 1]
   # add tokens to list
   texts.append(final_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
#print(dictionary.doc2id.x(texts[0],unknown_word_index=-1))

corpus=[]
for text in texts:
   corpus.append(dictionary.doc2idx(text, unknown_word_index=-1))

word_map = {}
for di, doc in enumerate(corpus):
    for wi, word in enumerate(doc):
        if(word in word_map):
            word_map[word].append((di,wi))
        else:
            word_map[word] = [(di,wi)]

K = 20
V = len(dictionary)
print(V)
D = len(corpus)
alpha = np.ones(K)*0.5
beta = np.random.rand(K, V)
for k in range(K):
   beta[k, :] = beta[k, :] / np.sum(beta[k, :])

print(D)

phi_init = []
gamma_init = []
for d in range(D):
   N = len(corpus[d])  # number of words in the document
   phi_init.append(np.ones((N, K)) * 1 / K)
   gamma_init.append(np.ones(K) * (alpha + N / K))

count = 0
convergence = 0
while convergence == 0:
   print("e_step")
   [phi, gamma] = E_step(corpus, K, D, alpha, beta, phi_init, gamma_init)
   print("m_step")
   [alpha, beta] = M_step(alpha, beta, phi, gamma, corpus, D, V, K)

   count += 1

   if(count % 5 == 0):
        print_results_beta(beta, dictionary, K, V, 10)

        print("*******************************************************")

        print_results(phi, corpus, dictionary, K, D, 10)

        print("*******************************************************")

   if count > 50:
       convergence = 1











