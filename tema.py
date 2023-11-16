import pandas as pd
import numpy as np
import math as mt
import json


train_data = pd.read_csv("Heart_attack.csv")
train_data = train_data.dropna()

#mean_values = train_data.mean()
#variance_values = train_data.var()

#print("Mean values:\n", mean_values)
#print("\nVariance values:\n", variance_values)

def compute_probabilities(train_data, attribute):
   
    pmf = train_data[attribute].value_counts(normalize=True)
    return pmf


target_attribute = 'target'
discrete_attributes = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
continuous_attributes=['age','trestbps','chol','thalach','oldpeak']

"""
for attribute in discrete_attributes:
    pmf_result = compute_probabilities(train_data, attribute)
    print(f"Probability Mass Function for {attribute}:\n", pmf_result)
"""
def calculate_entropy(train_data,attribute):
    entropy_result=0
    attribute_pmf=compute_probabilities(train_data,attribute)
    probs=attribute_pmf.values
    probs_list=probs.tolist()
    for prob in probs_list:
        entropy_el = - prob*np.log2(prob)
        entropy_result+=entropy_el
    return entropy_result

#print(calculate_entropy(train_data,"target"))
#H(TARGET_ATTRIBUTE|ATTRIBUTE)
def calculate_condition_entropy(train_data,target_attribute,attribute):
    attribute_pmf = compute_probabilities(train_data,attribute)
    conditional_entropy = 0
    for value in attribute_pmf.index:
        subset = train_data[train_data[attribute] == value]
        subset_probability = attribute_pmf[value]
        subset_entropy = calculate_entropy(subset,target_attribute)
        conditional_entropy += subset_probability * subset_entropy

    return conditional_entropy



def calculate_information_gain(train_data,target_attribute,attribute):
    return calculate_entropy(train_data,target_attribute)-calculate_condition_entropy(train_data,target_attribute,attribute)
""""
for att in discrete_attributes:
   print(f"the other attribute is {att} and the IG is: {calculate_information_gain(train_data,'target',att)}")
"""
def find_root_node(train_data,discrete_list,target):
    max_information_gain=0
    maxim_attribute='target'
    for attribute in discrete_list:
        information_gain=calculate_information_gain(train_data,target,attribute)
        if max_information_gain<=information_gain:
            max_information_gain=information_gain
            maxim_attribute=attribute
    return (maxim_attribute,max_information_gain)

#print(find_root_node(train_data,discrete_attributes,'target'))

def get_splits(continuous_attribute,labels):
    train_data = pd.DataFrame({"Attribute": continuous_attribute, "Label": labels})
    train_data = train_data.sort_values(by="Attribute")
    splits=[]
    for value in range(1,len(train_data)):
        if train_data["Label"].iloc[value] != train_data["Label"].iloc[value - 1]:
            split_point = (train_data["Attribute"].iloc[value] + train_data["Attribute"].iloc[value - 1]) / 2.0
            splits.append(split_point)
    return splits

#print(get_splits(train_data["age"],train_data["target"]))  

def id3_discrete(train_data, attributes, target_attribute, parent_node_class=None):
    if len(train_data[target_attribute].unique()) == 1:
        return train_data[target_attribute].iloc[0]
    if len(attributes) == 0:
        return parent_node_class
    best_feature, info_gain = find_root_node(train_data, attributes, target_attribute)
    tree = {"node_attribute": best_feature, "n_observations": dict(train_data[target_attribute].value_counts()), "information_gain": info_gain}
    for value in train_data[best_feature].unique():
        subset = train_data[train_data[best_feature] == value]
        remaining_features = attributes.copy()
        remaining_features.remove(best_feature)
        subtree = id3_discrete(subset, remaining_features, target_attribute, train_data[target_attribute].mode().iloc[0])
        tree["values"] = tree.get("values", {})
        tree["values"][value] = subtree
    return tree

def id3(train_data,discrete_attributes,continuous_attributes,target_attribute,parent_node_class=None):
    if len(train_data[target_attribute].unique()) == 1:
        return train_data[target_attribute].iloc[0]
    if len(discrete_attributes) == 0 and len(continuous_attributes) == 0:
        return parent_node_class
    best_feature, split_point, info_gain = find_root_node_dc(train_data, discrete_attributes,continuous_attributes,target_attribute)
    tree = {"node_attribute": best_feature, "n_observations": dict(train_data[target_attribute].value_counts()), "information_gain": info_gain}
    if split_point is not None:
        subset1 = train_data[train_data[best_feature] <= split_point]
        subset2 = train_data[train_data[best_feature] > split_point]

        remaining_discrete_attributes = discrete_attributes.copy()
        remaining_continuous_attributes = continuous_attributes.copy()

        if best_feature in remaining_discrete_attributes:
            remaining_discrete_attributes.remove(best_feature)
        elif best_feature in remaining_continuous_attributes:
            remaining_continuous_attributes.remove(best_feature)

        tree["values"] = {
            f"less_than_or_equal_to {split_point}": id3(subset1, remaining_discrete_attributes, remaining_continuous_attributes, target_attribute, train_data[target_attribute].mode().iloc[0]),
            f"greater_than {split_point}": id3(subset2, remaining_discrete_attributes, remaining_continuous_attributes, target_attribute, train_data[target_attribute].mode().iloc[0])
        }
    else:
        for value in train_data[best_feature].unique():
            subset = train_data[train_data[best_feature] == value]
            remaining_discrete_attributes = discrete_attributes.copy()
            remaining_continuous_attributes = continuous_attributes.copy()

            if best_feature in remaining_discrete_attributes:
                remaining_discrete_attributes.remove(best_feature)
            elif best_feature in remaining_continuous_attributes:
                remaining_continuous_attributes.remove(best_feature)

            subtree = id3(subset, remaining_discrete_attributes, remaining_continuous_attributes, target_attribute, train_data[target_attribute].mode().iloc[0])
            tree["values"] = tree.get("values", {})
            tree["values"][value] = subtree

    return tree
    
def calculate_information_gain_discrete_continous(train_data, target_attribute, attribute, subset1, subset2):
    total_entropy = calculate_entropy(train_data, target_attribute)

    prob1 = len(subset1) / len(train_data)
    prob2 = len(subset2) / len(train_data)

    entropy_subset1 = calculate_entropy(subset1, target_attribute)
    entropy_subset2 = calculate_entropy(subset2, target_attribute)
    return total_entropy - (prob1 * entropy_subset1 + prob2 * entropy_subset2)

def find_best_split_point(train_data, target_attribute, attribute, splits):
    max_info_gain = -1
    best_split_point = None

    for split_point in splits:
        subset1 = train_data[train_data[attribute] <= split_point]
        subset2 = train_data[train_data[attribute] > split_point]

        info_gain = calculate_information_gain_discrete_continous(train_data, target_attribute, attribute, subset1, subset2)

        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_split_point = split_point

    return best_split_point, max_info_gain

def find_root_node_dc(train_data, discrete_attributes, continuous_attributes, target_attribute):
    best_feature = None
    split_point = None
    max_info_gain = -1

    for attribute in discrete_attributes:
        info_gain = calculate_information_gain(train_data, target_attribute, attribute)

        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = attribute
            split_point = None

    for attribute in continuous_attributes:
        splits = get_splits(train_data[attribute], train_data[target_attribute])
        current_split_point, info_gain = find_best_split_point(train_data, target_attribute, attribute, splits)

        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = attribute
            split_point = current_split_point

    return best_feature, split_point, max_info_gain

decision_tree_combined = id3(train_data, discrete_attributes, continuous_attributes, target_attribute)
def convert_np_int64(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {convert_np_int64(key): convert_np_int64(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_int64(element) for element in obj]
    return obj

def write_tree_to_file(tree, file_path):
    tree_converted = convert_np_int64(tree)
    with open(file_path, 'w') as file:
        json.dump(tree_converted, file, indent=4)
        
write_tree_to_file(decision_tree_combined,"id3.json")


decision_tree = id3_discrete(train_data, discrete_attributes, target_attribute)
#print(decision_tree)
write_tree_to_file(decision_tree,"id3_discrete.json")

with open('id3.json', 'r') as fisier:
    data = json.load(fisier)
print(json.dumps(data, indent=2))
        
