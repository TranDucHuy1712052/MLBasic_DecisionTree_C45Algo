import math
# ===========================================

# This class read data and pre-process them.
#   data:       list of row. Each row contains attributes' values and a class (final result)
#   classes:    list of classes' name
#   attName:    list of attributes' name
#   attVal:     list of attributes' values. This is a 2D array
#   attCont:    type of attributes (continous or discrete?)
class DataPackage:

    def __init__(self, dataPath, despPath):
        self.dataPath = dataPath
        self.despPath = despPath
        self.data = []
        self.classes = []
        self.attName = []
        self.attVal = {}
        self.attCont = []

    def readData(self):

        print("Reading data...")
        # read class and atts
        with open(self.despPath, "r") as file:
            # first row is list of classes
            classes = next(file)
            self.classes = [x.strip() for x in classes.split(",")]
            print(self.classes)
            # first 2 params of each row is name and type of attribute
            # - continous: "<name> : true"
            # - discrete: "<name> : false". Add 1 more line to contain values
            for line in file:
                [name, isCont] = [x.strip() for x in line.split(":")]
                isCont = bool(isCont)
                self.attName.append(name)          
                self.attCont.append(isCont)

                if (not isCont):
                    valLine = next(file)             
                    values = [valLine.strip() for x in valLine.split(",")]
                    self.attVal[name] = values
                else:
                    self.attVal[name] = []             # it's continous, does not need a specific value

        print("Names", self.attName)
        print("Values", self.attVal)
        print("isContinous", self.attCont)

        # read data file
        with open(self.dataPath, "r") as file:
            for line in file:
                row = [x.strip() for x in line.split(",")]
                if row != [] and row != [''] and (len(row) != 0):
                    self.data.append(row)
        print("\n ----------- \n")

    
    # transfer all data into float (if continous)
    def preprocess(self):
        for i in range(len(self.data)):
            for att_idx in range(len(self.attName)):
                if (self.attCont[att_idx]):
                    #print(self.data[i])
                    self.data[i][att_idx] = float(self.data[i][att_idx])
                    

# ====================================================

# Implement c45
class DecisionTree_C45:

    def __init__(self, DataPackage):
        self.data_package = DataPackage     
        self.rootNode = None                

    # ===== HELP FUNCTIONS =====

    # ID of 1 class name
    def class_idx(self, classname):
        i = 0
        for x in self.data_package.classes:
            if (x == classname):
                return i
            i += 1
        return -1

    # ID of 1 attribute name
    def attribute_idx(self, name):
        i = 0
        for x in self.data_package.attName:
            if (x == name):
                return i
            i += 1
        return -1

    # Entropy of data
    def entropy(self, data):
        
        classCount = [0 for x in self.data_package.classes]
        n = len(data)              

        for row in data:
            idx = self.class_idx(row[-1])
            classCount[idx] += 1

        en = float(0.0)
        # pi * log2(pi) ; pi = attCount[i] / N
        for c in classCount:
            p = float( float(c) / n)            # must convert to float
            if (p != 0):
                en += (-1.0) * p * math.log(p, 2)

        return en


    # Entropy with subsets
    # subsets = data, but divided into smaller pieces
    # VD: data = [1,2,3,4]; subsets = [ [1,2] , [3,4] ]
    def Entropy_with_subset(self, data, subsets):
        n = len(data)
        weights = [float( float(len(subset)) / n) for subset in subsets]
        e = 0
         # entropy of each subset
        for i in range(len(weights)):
            e += weights[i] * self.entropy(subsets[i])          
        return e



    # =========================================================
    # =================== MAIN FUNCTIONS ======================
    # =========================================================

    # find best attribute
    # atts = list of the attributes that are not processed
    def Split_data(self, data, atts):
        if (len(atts) == 0):
            return None

        subsets = []
        best_entropy = float("inf")            ## smallest entropy (default: inf)
        best_att = None            
        best_threshold = None                   ## only for continous attribute

        # scan all attributes
        for att in atts:

            # check if this is continous?
            att_idx = self.attribute_idx(att)
            isCont = self.data_package.attCont[att_idx]

            if (isCont):
                ## Sort this out
                data.sort(key = lambda row: row[att_idx])

                ## We consider each pair of rows: Get their average, and then split data into halves (smaller, larger than threshold)
                smallerData = []
                largerData = []
                threshold = 0
                n = len(data)
                for i in range(0, len(data) - 1):
                    if (data[i][att_idx] == data[i+1][att_idx]):
                        continue
                    threshold = (data[i][att_idx] + data[i+1][att_idx] ) / 2
                    smallerData = data[0:i+1]         
                    largerData = data[i+1:]      

                    tmp_subsets = [smallerData, largerData]
                    e = self.Entropy_with_subset(data, tmp_subsets)
                    
                
                    if (e < best_entropy):      
                        best_entropy = e
                        subsets = tmp_subsets
                        best_threshold = threshold
                        best_att = att
                        
            else:
                ## discrete attribute
                attValues = self.data_package.attVal[att_idx]
                tmp_subsets = [[] for x in attValues]

                for row in data:
                    for i in range(len(attValues)):
                        if (row[att_idx] == attValues[i]):
                            tmp_subsets[i].append(row)          
                            break
                e = self.Entropy_with_subset(data, tmp_subsets)
                if (e < best_entropy):
                        best_entropy = e
                        subsets = tmp_subsets
                        best_threshold = threshold
                        best_att = att
        return (best_att, best_entropy, best_threshold, subsets)


    def OnlyOneClass(self, data):
        for row in data:
            if (row[-1] != data[0][-1]):
                return False
        return True


    ## ================= BUILD TREE ====================

    # build tree using recursion, return a node to add into parent node's children list.
    def Recursive_Build_Tree(self, data, atts):
        
        if (len(data) == 0) :
            return
        # only one class left => Leaf node
        elif (self.OnlyOneClass(data)):
            className = data[0][-1]
            node = Node(className, True, 0)        
            return node
        # no attributes left but still not really clean?
        elif (len(atts) == 0) :
            node = Node("Mixed", True, 0)
        else:
            # find best attribute
            [best_att, best_entropy, best_threshold, subsets] = self.Split_data(data, atts)
           ## print("Attribute : ", best_att, " with e = ", best_entropy, " and threshold = ", best_threshold)                       #debug
            remainingAtts = atts[:]
            remainingAtts.remove(best_att)          
            node = Node(best_att, False, best_threshold)
            # for each subset => new child
            node.children = [ self.Recursive_Build_Tree(subset, remainingAtts) for subset in subsets]
            return node


    # build this tree based on data imported before
    def Build_Tree(self):
        print("Start building tree...")
        self.rootNode = self.Recursive_Build_Tree(self.data_package.data, self.data_package.attName)


    ## ================= OUTPUT ====================
    def Print_Tree(self):
        print("Decision tree : ")
        self.Recursive_Print_Tree(self.rootNode, 0)
        
    def Recursive_Print_Tree(self, node, depth):
        if (not node):
            return
        node.printNode(depth)
        for child in node.children:
            self.Recursive_Print_Tree(child, depth + 1)


class Node:
    def __init__ (self, name, isLeaf, threshold):
        self.name = name
        self.isLeaf = isLeaf
        self.threshold = threshold
        self.children = []
    
    def printNode(self, depth):
        tmp_str = ""
        for i in range(depth):
            tmp_str += "+"    
        print(tmp_str, "Name = {0}, isLeaf = {1}, threshold = {2}".format( self.name, self.isLeaf, self.threshold))






## MAIN 

data = DataPackage("./data/iris/iris.data", "./data/iris/iris.desp")
data.readData()
data.preprocess()

tree = DecisionTree_C45(data)
tree.Build_Tree()
tree.Print_Tree()
