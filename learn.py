from sklearn import tree
from sklearn import ensemble
from sklearn.externals.six import StringIO
import random, math
import numpy as np

####################
#####   abstract classes
####################

# abstract base class of the algorithms
class algorithm:
    def train(self, x, y):    pass
    def predict(self, x):     pass
    def compare(self, x, y):  pass
    def test(self, x, y):
        p = self.predict(x)
        score = 0
        for i in range(len(p)):
            if self.compare(p[i], y[i]):
                score += 1
        return score / len(p)

# base of the classifier algorithms
class algorithmc(algorithm):
    def compare(self, x, y):
        return x == y

# base of the regression algorithms
class algorithmr(algorithm):
    def compare(self, x, y):
        return abs(x - y) <= .75

####################
#####   decision tree
####################

class dtree():
    def __init__(self, depth, sampsplit, sampleaf):
        self.d = depth
        self.sampsplit = sampsplit
        self.sampleaf = sampleaf
        self.tree = None
    def predict(self, x):
        if self.tree == None:
            raise Exception('Model hasn\'t been trained yet!')
        return self.tree.predict(x)
    def draw(self, file):    # create *.dot file
        if self.tree == None:
            raise Exception('Model hasn\'t been trained yet!')
        with open(file, 'w') as f:
            tree.export_graphviz(self.tree, out_file = f)

class dtreec(dtree, algorithmc):
    def train(self, x, y):
        dtc = tree.DecisionTreeClassifier(max_depth=self.d,
            min_samples_split=self.sampsplit, min_samples_leaf=self.sampleaf)
        self.tree = dtc.fit(x,y)

class dtreer(dtree, algorithmr):
    def train(self, x, y):
        dtr = tree.DecisionTreeRegressor(max_depth=self.d,
            min_samples_split=self.sampsplit, min_samples_leaf=self.sampleaf)
        self.tree = dtr.fit(x,y)

####################
#####   custom decision tree
####################

class treenode:
    def __init__(self, dataset, availattrs, parent):
        self.attrs = availattrs
        self.dataset = dataset.copy()
        self.parent = parent
        self.isleaf = False
        # C4.5 algorithm
        if self.__isterminal():
            self.isleaf = True
            self.__setvalue()
        else:
            self.entropy = self.calcentropy(self.dataset)
            self.__createchildren()
    
    def calcentropy(self, dataset):
        if len(dataset) == 0: return 0
        neg = pos = 0
        for data in dataset:
            if data[-1] == 1:
                pos += 1
        neg = len(dataset) - pos
        pos /= len(dataset)
        neg /= len(dataset)

        lv = 0 if pos == 0 else math.log(pos,2)
        rv = 0 if neg == 0 else math.log(neg,2)
        s = pos * lv + neg * rv
        return -s
    
    def __calcmaxattrgain(self):
        maxgain = maxattr = -1000
        for attr in self.attrs:
            gain = self.__calcinfgain(attr)
            if gain > maxgain:
                maxgain = gain
                maxattr = attr
        return maxattr
    
    def __calcinfgain(self, attrindex):
        ld, rd, m = self.__splitdataset(attrindex)
        ls = len(ld) / len(self.dataset) * self.calcentropy(ld)
        rs = len(rd) / len(self.dataset) * self.calcentropy(rd)
        return self.entropy - (ls + rs)

    def __splitdataset(self, attr):
        attrvals = []
        for data in self.dataset:
            attrvals.append(float(data[attr]))
        m = np.mean(attrvals)   # use mean as threshold
        lcdata = []
        rcdata = []
        for data in self.dataset:
            if float(data[attr]) <= m:
                lcdata.append(data)
            else:
                rcdata.append(data)
        return lcdata, rcdata, m
    
    def __createchildren(self):
        maxattr = self.__calcmaxattrgain()
        lcdata, rcdata, m = self.__splitdataset(maxattr)
        
        self.attr = maxattr
        self.value = m
        
        childattrs = self.attrs.copy()
        childattrs.remove(maxattr)
        self.leftchild = treenode(lcdata, childattrs, self)
        self.rightchild = treenode(rcdata, childattrs, self)     
        
    def __isterminal(self):
        if len(self.attrs) == 0: return True
        if len(self.dataset) == 0: return True 
        p = [0,0]
        for data in self.dataset:
            p[data[-1]] += 1
        if p.count(0) == 1: return True
        return False
    
    def __setvalue(self):
        ds = self.dataset
        if len(self.dataset) == 0 and not self.parent == None:
            ds = self.parent.dataset  # use parent's value if no data available
        # otherwise based on the dominant vote
        p = [0,0]
        for data in ds:
            p[data[-1]] += 1
        self.value = p.index(max(p))

class dtree_custom(algorithmc):
    def __modtrainset(self, x, y):
        ts = []
        for i in range(len(x)):
            ts.append(x[i] + [y[i]])
        return ts

    def fit(self, x, y):
        self.train(x, y)
        return self
    
    def train(self, x, y):
        ts = self.__modtrainset(x, y)
        self.root = treenode(ts, [a for a in range(len(x[0]))], None)

    def __matchdata(self, x):
        node = self.root
        while not node.isleaf:
            if float(x[node.attr]) <= node.value:
                node = node.leftchild
            else:
                node = node.rightchild
        return node.value
    
    def predict(self, x):
        y = []
        for data in x:
            y.append(self.__matchdata(data))
        return y

####################
#####   bagging
####################

class bagging:
    def __init__(self, depth, sampsplit, sampleaf, size, count):
        self.d = depth
        self.sampsplit = sampsplit
        self.sampleaf = sampleaf
        self.size = size
        self.count = count
        self.forest = None
    def train(self, x, y):
        self.forest = []
        for i in range(self.size):
            xx = []
            yy = []
            for j in range(self.count):
                index = random.randint(0, len(x) - 1)
                xx.append(x[index])
                yy.append(y[index])
            t = self.createtree()
            t = t.fit(xx, yy)
            self.forest.append(t)
    def createtree(self): pass

class baggingc(bagging, algorithmc):
    def createtree(self):
        return tree.DecisionTreeClassifier(max_depth=self.d,
            min_samples_split=self.sampsplit, min_samples_leaf=self.sampleaf)
    def predict(self, x):
        if self.forest == None:
            raise Exception('Model hasn\'t been trained yet!')
        yy = []
        for tree in self.forest:
          yy.append(tree.predict(x))
        y = []
        for i in range(len(x)):
            r = []
            for j in range(len(yy)):
                r.append(yy[j][i])
            y.append(r.count(1) >= r.count(0))
        return y

class baggingc_custom(baggingc):
    def __init__(self, size, count):
        super().__init__(0, 0, 0, size, count)
    def createtree(self):
        return dtree_custom()

class baggingr(bagging, algorithmr):
    def createtree(self):
        return tree.DecisionTreeRegressor(max_depth=self.d,
            min_samples_split=self.sampsplit, min_samples_leaf=self.sampleaf)
    def predict(self, x):
        if self.forest == None:
            raise Exception('Model hasn\'t been trained yet!')
        yy = []
        for tree in self.forest:
          yy.append(tree.predict(x))
        y = []
        for i in range(len(x)):
            m = 0
            for j in range(len(yy)):
                m += yy[j][i]
            m /= len(yy)
            y.append(m)
        return y

####################
#####   random forest
####################

class random_forest():
    def __init__(self, depth, sampsplit, sampleaf):
        self.d = depth
        self.sampsplit = sampsplit
        self.sampleaf = sampleaf
        self.forest = None
    def predict(self, x):
        if self.forest == None:
            raise Exception('Model hasn\'t been trained yet!')
        return self.forest.predict(x)

class random_forestc(random_forest, algorithmc):
    def train(self, x, y):
        rfc = ensemble.RandomForestClassifier(max_depth=self.d,
            min_samples_split=self.sampsplit, min_samples_leaf=self.sampleaf)
        self.forest = rfc.fit(x,y)

class random_forestr(random_forest, algorithmr):
    def train(self, x, y):
        rfr = ensemble.RandomForestRegressor(max_depth=self.d,
            min_samples_split=self.sampsplit, min_samples_leaf=self.sampleaf)
        self.forest = rfr.fit(x,y)

