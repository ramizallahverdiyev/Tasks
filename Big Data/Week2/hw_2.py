from mrjob.job import MRJob

class MRMaxMinVotersNoHeader(MRJob):

    def mapper(self, _, line):
        parts = line.strip().split(',')
        if len(parts) != 3:
            return
        city, time, voters_str = parts
        try:
            voters = int(voters_str)
            yield (city, time), voters
        except ValueError:
            pass

    def reducer(self, key, values):
        min_voters = None
        max_voters = None
        for v in values:
            if min_voters is None or v < min_voters:
                min_voters = v
            if max_voters is None or v > max_voters:
                max_voters = v
        yield key, (min_voters, max_voters)

if __name__ == '__main__':
    MRMaxMinVotersNoHeader.run()