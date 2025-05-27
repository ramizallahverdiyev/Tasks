from mrjob.job import MRJob

class VirusGotCount(MRJob):

    def mapper(self, _, line):
        date = line.split(',')[1].split()[0]
        degree = line.split(',')[1].split()[1]

        if float(degree) >= 37.0:
            yield date, 1

    def reducer(self, key, values):
        yield key, sum(values)

if __name__ == '__main__':
    VirusGotCount.run()