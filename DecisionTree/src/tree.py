'''
Created on Oct 15, 2017

@author: surya
'''

import math
from copy import deepcopy
from node import Node
from scipy.stats import chi2
import node
from lib2to3.fixer_util import Attr

class Tree(object):
    '''
    Represents the decision tree.
    '''

    tree = {}
    
    '''
    Learns the decision tree by fitting the training data.
    @param dataset: Training dataset
    @param confidence: Confidence value for chi-square threshold.
    '''
    def learn(self, dataset, confidence):
        self.tree = buildTree(dataset.instances, dataset.attributes, dataset.target_attr, confidence)
    
    '''
    Classifies the given set of test_instances.
    @param test_instances: Set of training instances.
    @param target_attr: Target attribute to predict.
    
    @return: List of predicted and actual class labels.
    '''
    def classify(self, test_instances, target_attr):
        predicted_vals = []
        actual_vals = []
        for instance in test_instances:
            predicted_val, actual_val = getTargetVal(self.tree, instance, target_attr), instance[target_attr.idx]
            predicted_vals.append(predicted_val)
            actual_vals.append(actual_val)
        
        return predicted_vals, actual_vals

'''
Method to classify the test_instance and return the class label.

@param test_instance: test instance.

@return: Class label for the given test instance.
'''      
def getTargetVal(node, test_instance, target_attr):
    if node.attr.idx == target_attr.idx:
        return node.targetVal
    else:
        instance_attr_val = test_instance[node.attr.idx]
        if instance_attr_val is None:
            instance_attr_val = node.mostFrequentAttrVal
            
        for child_node in node.children:
            if child_node.parent_attr_val == instance_attr_val:
                return getTargetVal(child_node, test_instance, target_attr)

'''
Builds the decision tree recursively by learning the best attribute at each node.

@param instaces: List of training instances at that node.
@param attributes: List of attributes considered for computing best attribute at the node.
@param confidence: Confidence factor used for computing the chi square threshold
@param target_attr: Target attribute in the data set.
@param parent_attr_val: Parent node's attribute value which leads to this node.
@param depth: Depth of the node in the tree.

@return: Returns the root node of the decision tree.
''' 
def buildTree (instances, attributes, target_attr, confidence, parent_attr_val = '', depth = 0): 
    
    initial_entropy, most_freq_class = getInitialEntropyAndMostFreqTargetVal(instances, target_attr)
    
    #If initial entropy is 0 (pure node) or if attributes list is empty, then return a node with the target value as the most_freq_class
    if initial_entropy == 0.0 or len(attributes) == 0:
        return Node(target_attr, most_freq_class, most_freq_class, parent_attr_val, depth)
    else:
        instances_with_missing_vals = deepcopy(instances)
        replaceMissingValues(instances_with_missing_vals, attributes, target_attr)
        best_attr, freq_val = getBestAttributeAndMostFreqAttrVal(instances_with_missing_vals, attributes, confidence, target_attr, initial_entropy)
        if best_attr is not None:
            print "Best attribute at depth %d: %s" % (depth, best_attr.name)
            attributes.remove(best_attr)
            
            #Create node with the best attribute
            node = Node(best_attr, freq_val, most_freq_class, parent_attr_val, depth)
            
            #Create child node for each of the best attribute values
            for attr_val in best_attr.values:
                instances_subset = []
                
                #Get the list of instances with this attribute value
                for idx in range(len(instances_with_missing_vals)):
                    if instances_with_missing_vals[idx][best_attr.idx] == attr_val:
                        
                        #Add the corresponding instance with missing values to the instances_subset
                        instances_subset.append(instances[idx])   
                
                #If there are no instances at this node, return a leaf node with the most_freq_class value from its parent node
                if len(instances_subset) == 0:
                    return Node(target_attr, most_freq_class, most_freq_class, parent_attr_val, depth)
                #Recursively build the subtree for the child node
                else:
                    node.children.append(buildTree(deepcopy(instances_subset), deepcopy(attributes), target_attr, confidence, attr_val, depth + 1))
                    
            return node
        
        #If no best attribute is found, return a leaf node with target value as the most frequent class label
        else:
            return Node(target_attr, most_freq_class, most_freq_class, parent_attr_val, depth)

'''
Replaces the missing values in the given set of training instances

@param instaces: List of training instances at that node.
@param attributes: List of attributes considered for computing best attribute at the node. 
@param target_attr: Target attribute in the data set.

@return: Returns the instances with the missing values replaced
''' 
def replaceMissingValues(instances, attributes, target_attr):
    for target_attr_val in target_attr.values:
        
        #Get subset of instances with target value == target_attr_val
        subset = [instance for instance in instances if instance[target_attr.idx] == target_attr_val]
        if len(subset) > 0:
            missing_attr_val = {}
            indices_of_attrs_to_remove = []
            for attr in attributes:
                all_values_in_subset = [instance[attr.idx] for instance in subset]
                majority_val = majorityAttrVal(all_values_in_subset)
                if majority_val is not None: 
                    missing_attr_val[attr.idx] = majority_val
                else:
                    #print "All values are missing in the subset for attribute: %s" % attr.name
                    all_values = [instance[attr.idx] for instance in instances]
                    majority_val = majorityAttrVal(all_values)
                    if majority_val is not None:
                        missing_attr_val[attr.idx] = majority_val
                    else:
                        #print "All values are missing in the entire set of instances for attribute: %s" % attr.name
                        #Add the attribute's index to list of indices_of_attrs_to_remove as all the values are missing for this attribute
                        indices_of_attrs_to_remove.append(attr.idx)
            
            #Filter out attributes with all values missing
            attributes = [attr for attr in attributes if not (attr.idx in indices_of_attrs_to_remove)]
            
            for instance in subset:
                for idx, attr_val in enumerate(instance):
                    if (idx in missing_attr_val) and (attr_val is None):
                        instance[idx] = missing_attr_val[idx]
 
'''
Finds the majority attribute value among the list of values

@param values: List of attribute values

@return: Highest frequency attribute value.
'''                        
def majorityAttrVal(values):
    freq = {}
    highest_freq = 0
    most_freq_val = values[0]
    for val in values:
        if val is not None:
            if freq.has_key(val):
                freq[val] += 1
            else:
                freq[val] = 1
            
            if freq[val] > highest_freq:
                highest_freq = freq[val]
                most_freq_val = val

    return most_freq_val

'''
Finds the best attribute and its most frequent value at the node for the list of instances

@param instaces: List of training instances at that node.
@param attr: Attribute for which the gain ratio is computed. 
@param target_attr: Target attribute in the data set.
@param initial_entropy: Initial class entropy of the instances. 

@return: Tuple containing the attribute's gain ratio and the most frequent attribute value at that node.
'''        
def getBestAttributeAndMostFreqAttrVal(instances, attributes, confidence, target_attr, initial_entropy):
    max_gain = 0.0
    best_attr = attributes[0]
    best_attr_freq_val = best_attr.values[0]
    for attr in attributes:
        gain_ratio, freq_val = getGainRatioAndMostFrequentAttrVal(instances, attr, target_attr, initial_entropy)
        if gain_ratio > max_gain:
            best_attr = attr
            best_attr_freq_val = freq_val
            max_gain = gain_ratio
    
    threshold = chi2.isf(1 - confidence, len(best_attr.values) - 1)
    chisquare = getChiSquareValue(instances, best_attr, target_attr)
    if chisquare > threshold:
        #print "Pass: best_attr: %s, chi-sq: %.3f, threshold: %.3f" % (best_attr.name, chisquare, threshold)
        return best_attr, best_attr_freq_val
    else:
        print "Fail: best_attr: %s, chi-sq: %.3f, threshold: %.3f" % (best_attr.name, chisquare, threshold)
        
    #Return None if no attribute satisfies the chi-square threshold.
    return None, None

'''
Computes the gain ratio and most frequent value of the given attribute for the list of instances

@param instaces: List of training instances at that node.
@param attr: Attribute for which the gain ratio is computed. 
@param target_attr: Target attribute in the data set.
@param initial_entropy: Initial class entropy of the instances. 

@return: Tuple containing the attribute's gain ratio and the most frequent attribute value at that node.
'''
def getGainRatioAndMostFrequentAttrVal(instances, attr, target_attr, initial_entropy):
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
    
    for attr_val in attr_val_freq.keys():
        subset = [instance for instance in instances if instance[attr.idx] == attr_val]
        subset_prob = len(subset)/float(total_instances)
        subsets_entropy += subset_prob * getEntropy(subset, target_attr)
        split_info += -subset_prob * math.log(subset_prob, 2)
        
    gain = (initial_entropy - subsets_entropy)
    gain_ratio = gain
    if split_info != 0.0:
        gain_ratio = gain/split_info
        
    #print attr.name, ": initial_entropy: %.3f, gain ratio: %.3f, gain = %.3f, split_info = %.3f" % (initial_entropy, gain_ratio, gain, split_info)
    return gain_ratio, most_freq_val

'''
Computes the entropy of the given list of instances for the target attribute

@param instaces: List of training instances at that node.
@param target_attr: Target attribute in the data set.

@return: Class entropy for the instances.
'''
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

'''
Computes the class entropy and the most frequency class value for the given set of instances

@param instaces: List of training instances at that node.
@param target_attr: Target attribute in the data set.

@return: Tuple containing class entropy and most frequent class value.
'''
def getInitialEntropyAndMostFreqTargetVal(instances, target_attr):
    class_freq = {}
    highest_freq = 0.0
    most_freq_target_val = target_attr.values[0]
    
    for instance in instances:
        target_val = instance[target_attr.idx]
        if class_freq.has_key(target_val):
            class_freq[target_val] += 1.0
        else:
            class_freq[target_val] = 1.0
        
        if class_freq[target_val] > highest_freq:
            highest_freq = class_freq[target_val]
            most_freq_target_val = target_val
        
    entropy = 0.0
    for freq in class_freq.values():
        prob = freq/len(instances)
        entropy += -prob * math.log(prob, 2.0)
    
    return entropy, most_freq_target_val

'''
Computes the chi-square value for the given attribute and threshold

@param instaces: List of training instances at that node.
@param attr: Attribute for which the value is computed.
@param target_attr: Target attribute in the data set.

@return: Chi-square value for the given attribute.
'''
def getChiSquareValue(instances, attr, target_attr):
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
            deviation += math.pow(target_val_instances_count - expected_count, 2)/(expected_count)

    return deviation
