# refer https://meenavyas.wordpress.com/2017/09/09/finding-similarity-between-text-documents/
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import nltk 

def process(file):
    raw = open(file).read()
    tokens = word_tokenize(raw)
    words = [w.lower() for w in tokens]

    porter = nltk.PorterStemmer() # or nltk.RegexpStemmer('ing$|s$|e$')
    stemmed_tokens = [porter.stem(t) for t in words]
    
    # Removing stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in stemmed_tokens if not w in stop_words]

    # count words
    count = nltk.defaultdict(int)
    for word in filtered_tokens: 
        count[word] += 1             
    return count;

def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def getSimilarity(dict1, dict2):
    all_words_list= []
    for key in dict1:
        all_words_list.append(key)
    for key in dict2:
        all_words_list.append(key)
    all_words_list_size = len(all_words_list)
        
    v1 = np.zeros(all_words_list_size, dtype=np.int)
    v2 = np.zeros(all_words_list_size, dtype=np.int)
    i = 0
    for (key) in all_words_list:
        v1[i] = dict1.get(key, 0)
        v2[i] = dict2.get(key, 0)
        i = i + 1
    return cos_sim(v1, v2);
    
if __name__ == '__main__':
    # python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
    dict1 = process("/home/aroot/Bug599831.txt")
    dict2 = process("/home/aroot/Bug800279.txt")
    dict3 = process("/home/aroot/Bug1055525.txt")
    print("Similarity between Bug#599831 and Bug#800279 is ", getSimilarity(dict1,dict2))
    print("Similarity between Bug#599831 and Bug#1055525 is ", getSimilarity(dict1,dict3))
    print("Similarity between Bug#800279 and Bug#1055525 is ", getSimilarity(dict2,dict3))
