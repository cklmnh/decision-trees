'''
Created on Oct 15, 2017

@author: surya
'''

import time
import sys
import arff
from dataset import Dataset
from tree import Tree
from collections import Counter

def main():
    sample_data_filepath = '../weather_data.arff'
    training_data_filepath = '../training_subsetD.arff'
    
    test_data_filepath = '../testingD.arff'
    
    sample_target_attr_name = 'play'
    target_attr_name = 'Class'
    
    confidence_values = [0, 0.5, 0.8, 0.95, 0.99]
    #confidence_values = [0.0]
    for confidence in confidence_values:
        output_file_name = "../HandleMissingValues_Method1/%.2f_result.txt" % confidence
        output_file = open(output_file_name, 'w')
        sys.stdout = output_file
        
        print "Confidence value: %.2f\n" % confidence
        start = time.clock()
        print "Reading input data..."
        #dataset = Dataset(sample_data_filepath, sample_target_attr_name)
        dataset = Dataset(training_data_filepath, target_attr_name)
        print "Successfully parsed training data in %.3fs." % (time.clock() - start)
        
        start = time.clock()
        dtree = Tree()
        print "Learning tree by fitting data...\n"
        dtree.learn(dataset, confidence)
        
        print "\nTree building successful in %.3fs." % (time.clock() - start)
        print "Number of decision nodes: %d" % dtree.decision_nodes_count
        print "\nPrinting tree..\n\n", dtree.tree.ToString()
        
        print "\n", "." * 120
        
        print "\nReading test data from test set..."
        start = time.clock()
        #test_data = arff.load(open(sample_data_filepath, 'rb'))
        test_data = arff.load(open(test_data_filepath, 'rb'))
        print "Successfully parsed test data in %.3fs." % (time.clock() - start)
        test_instances = test_data.get("data")
        print "Classifying test data instances...\n"
        classifyInstancesAndOutputPerfMetrics(dtree, test_instances, dataset.target_attr)
        
        print "\n", "." * 120
        
        print "\nReading test data from training set..."
        start = time.clock()
        #test_data = arff.load(open(sample_data_filepath, 'rb'))
        test_data = arff.load(open(training_data_filepath, 'rb'))
        test_instances = test_data.get("data")
        print "Successfully parsed test data in %.3fs." % (time.clock() - start)
        print "Classifying training data instances...\n"
        classifyInstancesAndOutputPerfMetrics(dtree, test_instances, dataset.target_attr)
        
        print "\n", "=" * 120, "\n"
        
        output_file.close()
'''
Classifies the given set of instances using the decision tree learned.

@param dtree: Decision tree created
@param instances: List of instances to classify
@param target_attr: Target attribute to predict
'''
def classifyInstancesAndOutputPerfMetrics(dtree, instances, target_attr):
    start = time.clock()
    predicted_values, actual_values, positive_len4_paths, negative_len4_paths = dtree.classify(instances, target_attr)
    print "Most common path (upto length 4) applied to the largest fraction of the positively-labeled training examples:\n", Counter(positive_len4_paths).most_common(1)
    print "Most common path (upto length 4) applied to the largest fraction of the negatively-labeled training examples:\n", Counter(negative_len4_paths).most_common(1)
    print "\nCompleted classification in %.3fs." % (time.clock() - start)
    
    true_pos, false_pos, true_neg, false_neg = getConfusionMatrix(predicted_values, actual_values, target_attr.values[0])
    accuracy = (true_pos + true_neg)/(len(instances))
    precision = 0 if true_pos == 0 else true_pos/(true_pos + false_pos)
    recall = 0 if true_pos == 0 else true_pos/(true_pos + false_neg)

    print "True positives: %.0f False positives: %.0f" % (true_pos, false_pos)
    print "False negatives: %.0f True negatives: %.0f" % (false_neg, true_neg)
    print "\nAccuracy: %.6f, Precision: %.6f, Recall: %.6f" % (accuracy, precision, recall)  
        
'''
Computes the confusion matrix from the predicted and actual class labels.

@param predicted_values: List of predicted class labels for the test data
@param actual_values: List of actual class labels for the test data 
@param true_val: True class value

@return Returns the tuple containing true positives, false positives, true negatives, false negatives and list of indices containing misclassified instances.
'''

def getConfusionMatrix(predicted_values, actual_values, true_val):
    tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0
    for idx in range(len(predicted_values)):
        if predicted_values[idx] == actual_values[idx]:
            if predicted_values[idx] == true_val:
                tp += 1.0
            else:
                tn += 1.0
        else:
            if predicted_values[idx] == true_val:
                fp += 1.0
            else:
                fn += 1.0
    
    return tp, fp, tn, fn

if __name__ == "__main__":
    main()
