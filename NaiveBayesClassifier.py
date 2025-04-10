import csv

class NaiveBayesClassifier:

    def __init__(self):
        self.train_data=[]
        self.test_data=[]
        self.stopwords=[]
        self.post_dict={}
        self.neg_dict={}
        self.post_n=0
        self.neg_n=0
        self.train_features=[]

    def extract_features(self, data, stopwords):
        feature_words=[]

        for i in range(1, len(data)):
            sentence=data[i][1]
            sentence=sentence.lower()

            for char in "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~":
                sentence=sentence.replace(char, " ")
                words=sentence.split()
                data[i][1]=words

            for word in words:
                feature_words.append(word)
        
        feature_rem_stopwords=[]

        for i in range(len(feature_words)):
            if feature_words[i] not in stopwords:
                feature_rem_stopwords.append(feature_words[i])
        
        freq_dict={}
        for word in set(feature_rem_stopwords):
            freq_dict[word]=0
        for word in feature_rem_stopwords:
            if word in freq_dict:
                freq_dict[word]+=1
        
        freq_dict=sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)

        final_features=freq_dict[:1000]
        for i in range(len(final_features)):
            final_features[i]=final_features[i][0]
        return final_features

    def preprocess(self, data, features):
        for i in range(1, len(data)):
            string=""
            for j in range(len(data[i][1])):
                if data[i][1][j] in features:
                    string=string+data[i][1][j]+" "
            data[i][1]=string
        return data

    def prob(self, sentence, freq_dict, n):
        words=sentence.split()
        prob=1
        for word in words:
            if word in freq_dict:
                prob=prob*(freq_dict[word]/n)
            else:
                prob=prob*(1/(n+len(freq_dict)))
        return prob

    def fit(self, train_data, stopwords):
        self.stopwords=stopwords
        self.train_data=train_data
        self.train_features=self.extract_features(self.train_data, self.stopwords)
        self.train_data=self.preprocess(self.train_data, self.train_features)
        
        for feature in self.train_features:
            self.post_dict[feature]=0
            self.neg_dict[feature]=0
        
        for i in range(1, len(self.train_data)):
            if self.train_data[i][0]==1:
                for word in self.train_data[i][1].split():
                    self.post_dict[word]+=1
                    self.post_n+=1
            elif self.train_data[i][0]==0:
                for word in self.train_data[i][1].split():
                    self.neg_dict[word]+=1
                    self.neg_n+=1

        return self.train_features

    def predict(self, test_data):
        self.test_data=test_data
        test_features=self.extract_features(self.test_data, self.stopwords)
        self.test_data=self.preprocess(self.test_data, self.train_features)
        
        predictions=[]
        for i in range(1, len(self.test_data)):
            post_p=self.prob(self.test_data[i][1], self.post_dict, self.post_n)
            neg_p=self.prob(self.test_data[i][1], self.neg_dict, self.neg_n)
            if post_p>=neg_p:
                predictions.append(1)
            else:
                predictions.append(0)
        
        return predictions
    
