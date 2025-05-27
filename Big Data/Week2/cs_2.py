from mrjob.job import MRJob

class Suit_Count(MRJob):

    def mapper(self, _, line):
        card_type = line.split(',')[1]
        yield card_type, 1

    def reducer(self, card_type, values):
        yield card_type, sum(values)

if __name__ == '__main__':
    Suit_Count.run()
    