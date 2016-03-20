import json, operator, time
from collections import defaultdict
from sklearn import tree
from sklearn import ensemble
from sklearn.externals.six import StringIO
from learn import *

# bunch class for movies

class movie:
    def __init__(self):
        self.__dict__ = dict()

    def tolist(self):
        return [self.runtime, self.season, self.rating,
                self.budget, self.genre, self.dir,
                self.plotfreqscore, self.actorscore]

####################
#####   data preprocessing
####################

# helper functions

def getruntime(runtime):
    t = runtime.split()[0]
    if t == 'N/A': return '120'
    return t

def getseason(date):
    s = int(date.split('-')[1])
    if s > 2 and s <= 5:    return 1
    elif s > 5 and s <= 8:  return 2
    elif s > 8 and s <= 11: return 3
    else:                   return 4

def getrating(rating):
    r = ['g','pg','pg-13','r','nc-17']
    rating = rating.lower()
    if rating in r: return r.index(rating)
    return len(r)

genrecode = dict()
genrecode['action']          = 1 << 0
genrecode['adventure']       = 1 << 1
genrecode['documentary']     = 1 << 2
genrecode['drama']           = 1 << 3
genrecode['family']          = 1 << 4
genrecode['fantasy']         = 1 << 5
genrecode['science fiction'] = 1 << 6
genrecode['romance']         = 1 << 7
genrecode['comedy']          = 1 << 8
genrecode['thriller']        = 1 << 9

def getgenrecode(genres):
    code = 0
    for g in genres:
        genre = g['name'].lower()
        if not genre in genrecode: continue
        code |= genrecode[genre]
    return code

# and the actual processing
        
raw = json.load(open('data.json'))

'''
# first, perform a simple lexical analysis
freqtab = defaultdict(int)
for m in raw:
    plot = m['Plot']
    for w in plot.split():
        if len(w) <= 3: continue
        freqtab[w.lower()] += float(m['imdbRating'])

freq_s = sorted(freqtab.items(),
                key=operator.itemgetter(1))
for item in freq_s[-200:]:
    print('{}:{}'.format(item[0],item[1]))
'''
# because of the filler words we chose the top 20 nouns:
freqtab = ['life','young','love','world','family','years','time','story',
           'friends','battle','people','team','woman','police','journey',
           'girl','home','night','father','drug']

# compute actor score
actscore = defaultdict(int)
actfreq = defaultdict(int)
for m in raw:
    actors = m['Actors'].split(',')
    for actor in actors:
        actor = actor.strip()
        actfreq[actor] += 1
        actscore[actor] += float(m['imdbRating'])
for actor in actscore.keys():
    actscore[actor] /= actfreq[actor]

dir_id = dict()
data = []
count = 0

for m in raw:
    mov = movie()
    
    mov.runtime = getruntime(m['Runtime'])
    mov.season = getseason(m['release_date'])
    mov.rating = getrating(m['Rated'])
    mov.budget = m['budget']
    mov.genre = getgenrecode(m['genres'])
    
    if not m['Director'] in dir_id.keys():
        dir_id[m['Director']] = len(dir_id.keys())
    mov.dir = dir_id[m['Director']]

    pfscore = 0
    for w in m['Plot'].split():
        if w in freqtab: pfscore += 1
    mov.plotfreqscore = pfscore

    acs = 0
    actors = m['Actors'].split(',')
    for actor in actors:
        acs += actscore[actor]
    acs /= len(actors)
    mov.actorscore = acs

    mov.score = float(m['imdbRating'])
    mov.fresh = int(float(m['imdbRating']) >= 6.0)
    
    data.append(mov)

####################
#####   training/testing
####################

random.shuffle(data)    # let's randomize the data

x = []
yf = []
ys = []
for d in data:
    x.append(d.tolist())
    yf.append(d.fresh)
    ys.append(d.score)

tc = round(len(data) * .65)
trainx = x[0:tc]
trainyf = yf[0:tc]
trainys = ys[0:tc]
testx = x[tc:]
testyf = yf[tc:]
testys = ys[tc:]


print('custom dtree\n')
cust = dtree_custom()
cust.train(trainx, trainyf)
s = cust.test(trainx, trainyf)
print('custom tree accuracy - train: {}'.format(s * 100))
s = cust.test(testx, testyf)
print('custom tree accuracy - test: {}'.format(s * 100))
print('*'*50)

print('scikit dtree\n')
for depth in [1000, 500, 50, 25, 8, 4]:
    for minsample in [1, 5, 25, 50, 500]:
        tc = dtreec(depth, minsample, minsample)
        tc.train(trainx, trainyf)
        sf = tc.test(testx, testyf)
        sft = tc.test(trainx, trainyf)

        tr = dtreer(depth, minsample, minsample)
        tr.train(trainx, trainys)
        sr = tr.test(testx, testys)
        srt = tr.test(trainx, trainys)
        print('depth={} minsample={}  scoretrain:{} / {}  scoretest: {} / {}'.format(depth,
            minsample, sft*100, srt*100, sf*100, sr*100))
        '''if depth == 4 and minsample == 50:
            tc.draw('treec.dot')
        if depth == 1000 and minsample == 25:
            tc.draw('treeunlim.dot')
        if depth == 1000 and minsample == 1:
            tc.draw('treeunlim.dot')'''
        
print('*'*50)

print('bagging custom trees\n')
for size in range(500,1501,500):
    for samples in range(500,1501,500):
        bcc = baggingc_custom(size, samples)
        bcc.train(trainx, trainyf)
        sctr = bcc.test(trainx, trainyf)
        scte = bcc.test(testx, testyf)
        print('size/count:{}/{} train:={} test={}'.format(size, samples,
                    sctr*100, scte*100))
print('*'*50)

print('bagging scikit trees\n')
for size in range(500,1501,500):
    for samples in range(500,1501,500):
        for depth in [1000, 500, 50, 25, 8, 4]:
            for minsample in [1, 5, 25, 50, 500]:
                bc = baggingc(depth, minsample, minsample, size, samples)
                bc.train(trainx, trainyf)
                sf = bc.test(testx, testyf)
                sft = bc.test(trainx, trainyf)

                br = baggingr(depth, minsample, minsample, size, samples)
                br.train(trainx, trainys)
                sr = br.test(testx, testys)
                srt = br.test(trainx, trainys)
                print('size/count:{}/{} depth={} minsample={}  scoretrain:{} / {}  scoretest: {} / {}'.format(size, samples, depth,
                    minsample, sft*100, srt*100, sf*100, sr*100))
print('*'*50)

print('random forests\n')
for depth in [1000, 500, 50, 25, 8, 4]:
    for minsample in [1, 5, 25, 50, 500]:
        fc = random_forestc(depth, minsample, minsample)
        fc.train(trainx, trainyf)
        sf = fc.test(testx, testyf)
        sft = fc.test(trainx, trainyf)

        fr = random_forestr(depth, minsample, minsample)
        fr.train(trainx, trainys)
        sr = fr.test(testx, testys)
        srt = fr.test(trainx, trainys)
        print('depth={} minsample={}  scoretrain:{} / {}  scoretest: {} / {}'.format(depth,
            minsample, sft*100, srt*100, sf*100, sr*100))
print('*'*50)
