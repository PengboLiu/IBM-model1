import codecs
from nltk import word_tokenize
from operator import itemgetter
from collections import defaultdict
import os

class IBMModel1:
    def __init__(self, num_sentences=20000):
        self.corpus = self.initialize_corpus(num_sentences)
        self.trans_prob = self.initialize_translation_probability()
        self.conditional_dict = defaultdict(list)

    def initialize_corpus(self, num_sentences=10000):
        directory_name = input("Enter a directory name to be read: \n")
        file_list = []
        for file in os.listdir(directory_name):
            if file.endswith("en"):
                filename = directory_name + file
                file_list.insert(0, (codecs.open(filename, "r", "utf-8")))
            elif file.endswith("zh"):
                filename = directory_name + file
                file_list.insert(1, (codecs.open(filename, "r", "utf-8")))
        i = 0
        corpus = dict()
        while i < num_sentences:
            sentence1 = tuple(word_tokenize("NULL " + file_list[0].readline().strip("\n").strip("¡").strip("¿").lower()))
            sentence2 = tuple(word_tokenize("NULL " + file_list[1].readline().strip("\n").strip("¡").strip("¿").lower()))
            corpus[sentence1] = sentence2
            i += 1
        print("read data √")
        return corpus

    def initialize_translation_probability(self):

        num_f_words = len(set(f_word for (english_sent, foreign_sent) in self.corpus.items() for f_word in foreign_sent))
        trans_prob = defaultdict(lambda: float(1/num_f_words))
        return trans_prob

    def train_model(self, iteration_count=10):

        for i in range(iteration_count):
            count_e_given_f = defaultdict(float)
            total = defaultdict(float)
            sentence_total = defaultdict(float)
            for (english_sent, foreign_sent) in self.corpus.items():

                for e_word in english_sent:
                    for f_word in foreign_sent:
                        sentence_total[e_word] += self.trans_prob[(e_word, f_word)]
                for e_word in english_sent:
                    for f_word in foreign_sent:
                        count_e_given_f[(e_word, f_word)] += (self.trans_prob[(e_word, f_word)]/sentence_total[e_word])
                        total[f_word] += (self.trans_prob[(e_word, f_word)]/sentence_total[e_word])
            for (e_word, f_word) in count_e_given_f:
                self.trans_prob[(e_word, f_word)] = count_e_given_f[(e_word, f_word)]/total[f_word]
            print("epoch = "+str(i) + " √")

        return self.trans_prob

    def print(self, num_iterations=300):
        print("Alignment probabilities")
        print()
        print("{:<40}{:>40}".format("t(e|f)", "Value"))
        print("--------------------------------------------------------------------------------")
        iterations = 0
        for ((e_word, f_word), value) in sorted(self.trans_prob.items(), key=itemgetter(1), reverse=True):
            if iterations < num_iterations:
                print("{:<40}{:>40.2}".format("t(%s|%s)" % (e_word, f_word), value))
            else:
                break
            iterations += 1


if __name__ == "__main__":
    ibm = IBMModel1()
    ibm.train_model()
    ibm.print()