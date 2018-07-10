# Extractive text summarization comparison

In this we compare a few Python libaries for Extractive text summarization.  In Extractive text summarization, the algorithm selects the most representative sentences from the paragraph that can effectively summarize it.  Unlike Abstractive summarization, it does not paraphrase the text.  

We compare the following Python packages:
  
1. Sumy (with the following algorithms)
  * Luhn's algorithm
  * LSA (Latent Semantic Analysis)
  * TextRank
  * LexRank
  * SumBasic
  * KL Distance
  
2. Gensim's TextRank algorithm
3. PyTeaser/TextTeaser
4. PyTextRank


## Text summarization algorithms (quick overview)

Luhn's algorithm is the most basic:
1. Ignore Stopwords
2. Determine Top Words: The most often occuring words in the document are counted up.
3. Select Top Words: A small number of the top words are selected to be used for scoring.
4. Select Top Sentences: Sentences are scored according to how many of the top words they 
contain. The top N sentences are selected for the summary.

SumBasic uses a simple concept:
1. get word prob. p(wi) = ni/N (ni = no. of times word w exists, N is total no. of words)
2. get sentence score sj = sum_{wi in sj} p(wi)/|wi| (|wi| = no. of times wi comes in sj)
3. choose sj with highest score
4. update pnew(wi) = pold(wi)^2 for words in the chosen sentence (we want probability to include the same words to go down)
5. repeat until you reach desired no. of sentences

KL algorithm solves arg min_{S} KL(PD || PS) s.t. len(S) <= # sentences, where 
	KL = Kullback-Lieber divergence = sum_{w} PD(w)log(PD(w)/PS(w))
	PD = unigram word distribution of the entire document
	PS = unigram word distribution of the summary (optimization variable)

LexRank and TextRank use a PageRank kind of algorithm
1. Treat each sentence as the node in the graph
2. Connect all sentences to get a complete graph (a clique basically)
3. Find similarity between si and sj to get weight Mij of the edge conecting i and j
4. Solve the eigen value problem Mp = p for similarity matrix M.
5. L = 0.15 + 0.85Mp.  L gives the final score for each sentence.  Pick the top sentences
LexRank uses a tf-idf modified cosine similarity for M.  TextRank uses some other similarity metric

LSA uses a SVD based approach
1. Get the term-sentence matrix A (rows is terms, columns is sentences).  Normalize with term-frequency (tf) only
2. Do SVD; A = USV' (A=m x n, U=m x n, S=n x n, V=n x n)
SVD derives the latent semantic structure of sentences.  The k dimensional sub-space get the key k topics
of the entire text structure.  It's a mapping from n-dimensions to k
If a word combination pattern is salient and recurring in document, this
pattern will be captured and represented by one of the singular vectors. The magnitude of the
corresponding singular value indicates the importance degree of this pattern within the
document. Any sentences containing this word combination pattern will be projected along
this singular vector, and the sentence that best represents this pattern will have the largest
index value with this vector. As each particular word combination pattern describes a certain
topic/concept in the document, the facts described above naturally lead to the hypothesis that
each singular vector represents a salient topic/concept of the document, and the magnitude of
its corresponding singular value represents the degree of importance of the salient
topic/concept.
Based on this, summarization can be based on matrix V.  V describes an importance degree 
of each topic in each sentence. It means that the k'th sentence we choose has the largest 
index value in k'th right singular vector in matrix V.  An extension of this is using 
SV' as the score for each sentence

Gensim uses its own TextRank algorithm.  The algorithm is same as the 
LexRank algorithm described above.  Only difference - instead of cosine similarity
it uses a more sophisticated BM-25 similarity metric 

TextTeaser/PyTeaser uses basic summarization features and build from it. Those features are:
1. Title feature is used to score the sentence with the regards to the title. It is calculated 
as the count of words which are common to title of the document and sentence.
2. Sentence length is scored depends on how many words are in the sentence. TextTeaser defined 
a constant “ideal” (with value 20), which represents the ideal length of the summary, in terms 
of number of words. Sentence length is calculated as a normalized distance from this value.
3. Sentence position is where the sentence is located. I learned that introduction and conclusion 
will have higher score for this feature.
4. Keyword frequency is just the frequency of the words used in the whole text in the bag-of-words 
model (after removing stop words).

PyTextRank is another TextRank algorithm. It works in four stages, each feeding its output to the next
1. Part-of-Speech Tagging and lemmatization are performed for every sentence in the document.
2. Key phrases are extracted along with their counts, and are normalized.
3. Calculates a score for each sentence by approximating jaccard distance between the sentence and key phrases.
4. Summarizes the document based on most significant sentences and key phrases.


## Example text

We use the following example text for the aforementioned algorithms:

"Analytics Industry is all about obtaining the "Information" from the data. With the growing amount of data in recent years, that too mostly unstructured, it's difficult to obtain the relevant and desired information. But, technology has developed some powerful methods which can be used to mine through the data and fetch the information that we are looking for.  One such technique in the field of text mining is Topic Modelling. As the name suggests, it is a process to automatically identify topics present in a text object and to derive hidden patterns exhibited by a text corpus. Thus, assisting better decision making.  Topic Modelling is different from rule-based text mining approaches that use regular expressions or dictionary based keyword searching techniques. It is an unsupervised approach used for finding and observing the bunch of words, called topics, in large clusters of texts.  Topic Models are very useful for the purpose for document clustering, organizing large blocks of textual data, information retrieval from unstructured text and feature selection. For example, New York Times are using topic models to boost their user article recommendation engines. Various professionals are using topic models for recruitment industries where they aim to extract latent features of job descriptions and map them to right candidates. They are being used to organize large datasets of emails, customer reviews, and user social media profiles."


## Summarization results
