from mrjob.job import MRJob
from mrjob.step import MRStep
import string

class MRWordFrequencyCountSorted(MRJob):

    def steps(self):
        return [
            MRStep(
                mapper=self.mapper_get_words,
                combiner=self.combiner_count_words,
                reducer=self.reducer_count_words
            ),
            MRStep(
                reducer=self.reducer_sort_counts
            )
        ]

    def mapper_get_words(self, _, line):
        for word in line.strip().split():
            clean_word = word.strip(string.punctuation).lower()
            if clean_word:
                yield clean_word, 1

    def combiner_count_words(self, word, counts):
        yield word, sum(counts)

    def reducer_count_words(self, word, counts):
        yield None, (sum(counts), word)

    def reducer_sort_counts(self, _, count_word_pairs):
        for count, word in sorted(count_word_pairs, reverse=True):
            yield word, count

if __name__ == '__main__':
    MRWordFrequencyCountSorted.run()
