'''
Created on Oct 15, 2017

@author: surya
'''

import math
from node import Node
from scipy.stats import chi2

class Tree(object):
    '''
    Represents the decision tree.
    
    Attributes:
        root: Represents the root node.
        training_set: List of data samples used for training and building the tree.
        validation_set: List of data samples used for validating/testing the tree.
        train_test_split: Percentage of training samples to be set aside for validation.
    '''

    tree = {}
    
    def learn(self, dataset, confidence, train_test_split):
        self.train_test_split = train_test_split
        self.tree = buildTree(dataset.instances, dataset.attributes, confidence, dataset.target_attr)
        print self.tree.ToString()

def buildTree (instances, attributes, confidence, target_attr, parent_attr_val = '', depth = 0):
    initial_entropy, target_val = getEntropyAndTargetVal(instances, target_attr)
    if initial_entropy == 0.0:
        return Node(target_attr.name, [], target_val, target_val, parent_attr_val, depth)
    else:
        best_attr, freq_val = chooseBestAttributeAndMostFreqVal(instances, attributes, confidence, target_attr, initial_entropy)
        if best_attr is not None:
            print best_attr.name
            attributes.remove(best_attr)
            updated_attr_set = attributes[:]
            node = Node(best_attr.name, [], freq_val, None, parent_attr_val, depth)
            for attr_val in best_attr.values:
                subset = [instance for instance in instances if instance[best_attr.idx] == attr_val]
                node.children.append(buildTree(subset, updated_attr_set, confidence, target_attr, attr_val, depth + 1))
            return node
        else:
            return Node(target_attr.name, [], target_val, target_val, parent_attr_val, depth)
        
def chooseBestAttributeAndMostFreqVal(instances, attributes, confidence, target_attr, initial_entropy):
    max_gain = 0.0
    best_attr = attributes[0]
    best_attr_freq_val = best_attr.values[0]
    for attr in attributes:
        gain_ratio, freq_val = getGainRatioAndMostFrequentVal(instances, attr, target_attr, initial_entropy)
        if gain_ratio > max_gain:
            best_attr = attr
            best_attr_freq_val = freq_val
            max_gain = gain_ratio
    
    threshold = chi2.isf(1 - confidence, len(best_attr.values) - 1)
    if chiSquaredTest(instances, best_attr, target_attr, threshold):
        return best_attr, best_attr_freq_val
    
    return None
  
def getGainRatioAndMostFrequentVal(instances, attr, target_attr, initial_entropy):
    attr_val_freq = {}
    total_instances = len(instances)
    most_freq_val = attr.values[0]
    highest_freq = 0.0
    
    for instance in instances:
        attr_val = instance[attr.idx]
        if attr_val_freq.has_key(attr_val):
            attr_val_freq[attr_val] += 1.0
        else:
            attr_val_freq[attr_val] = 1.0
        
        if attr_val_freq[attr_val] > highest_freq:
            most_freq_val = attr_val
            highest_freq = attr_val_freq[attr_val]
            
    subsets_entropy = 0.0
    split_info = 0.0
    
    for attr_val in attr_val_freq:
        attrValProb = attr_val_freq[attr_val]/total_instances
        subset = [instance for instance in instances if instance[attr.idx] == attr_val]
        subsets_entropy += attrValProb * getEntropy(subset, target_attr)
        subset_prob = len(subset)/float(total_instances)
        split_info += -subset_prob * math.log(subset_prob, 2)
        
    gain = (initial_entropy - subsets_entropy)
    gain_ratio = gain/split_info
    #print (attr.name + ": %.3f" % gain_ratio)
    
    return gain_ratio, most_freq_val

def getEntropy(instances, target_attr):
    class_freq = {}
    
    for instance in instances:
        target_val = instance[target_attr.idx]
        if class_freq.has_key(target_val):
            class_freq[target_val] += 1.0
        else:
            class_freq[target_val] = 1.0
            
    entropy = 0.0
    for freq in class_freq.values():
        prob = freq/len(instances)
        entropy += -prob * math.log(prob, 2.0)
    
    return entropy

def getEntropyAndTargetVal(instances, target_attr):
    class_freq = {}
    highest_freq = 0.0
    most_freq_class = target_attr.values[0]
    
    for instance in instances:
        target_val = instance[target_attr.idx]
        if class_freq.has_key(target_val):
            class_freq[target_val] += 1.0
        else:
            class_freq[target_val] = 1.0
        
        if class_freq[target_val] > highest_freq:
            highest_freq = class_freq[target_val]
            most_freq_class = target_val
            
    entropy = 0.0
    for freq in class_freq.values():
        prob = freq/len(instances)
        entropy += -prob * math.log(prob, 2.0)
    
    return entropy, most_freq_class

def chiSquaredTest(instances, attr, target_attr, threshold):
    deviation = 0.0
    attr_target_matrix = {}
    class_freq = {}
    total_instances_count = len(instances)
    
    for instance in instances:
        attr_val = instance[attr.idx]
        target_val = instance[target_attr.idx]
        
        if class_freq.has_key(target_val):
            class_freq[target_val] += 1.0
        else:
            class_freq[target_val] = 1.0
            
        if not (attr_target_matrix.has_key(attr_val)):
            attr_target_matrix[attr_val] = {}
            
        if (attr_target_matrix[attr_val].has_key(target_val)):
            attr_target_matrix[attr_val][target_val] += 1.0
        else:
            attr_target_matrix[attr_val][target_val] = 1.0
    
    for attr_val in attr_target_matrix.keys():
        instances_count = sum(attr_target_matrix[attr_val].values())
        
        for target_val in target_attr.values:
            target_val_instances_count = attr_target_matrix[attr_val][target_val] if attr_target_matrix[attr_val].has_key(target_val) else 0.0 
            target_val_freq = class_freq[target_val]/total_instances_count if class_freq.has_key(target_val) else 0.0    
            expected_count = target_val_freq * instances_count 
            
            #print attr_val + ", " + target_val +  ", exp: %.1f" % expected_count
            deviation += math.pow(target_val_instances_count - expected_count, 2)/(expected_count)
            
    #print deviation    
    #print class_freq
    #print attr_target_matrix
    return deviation > threshold