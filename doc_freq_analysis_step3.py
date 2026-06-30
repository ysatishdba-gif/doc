from config import project_id, bucket, nb_results_json, shopping_list_team
from utils import read_json_from_gcs
from collections import Counter
import itertools
from copy import deepcopy
import re
import json

#### Dependencies to run this file #####
loinc_class_doctype_map_dict = "Your final loinc_class_doctype_map_dict variable result from running map_doctypes_to_norm_class_step1.py"
#### End of dependencies to run this file #####

# create doctype frequency analysis
def doctype_freq(data):
    all_doctypes = list(itertools.chain.from_iterable(data.values()))
    doctype_counts = Counter(all_doctypes)
    total_ques = len(data.keys())

    # --- NEW: Sort the items by count (the second element in the item tuple) ---

    sorted_counts_list = sorted(doctype_counts.items(), key=lambda item: item[1], reverse=True)

    # --- Create the final dictionary in the new sorted order ---
    sorted_doctype_stats = {}
    for doctype_name, count in sorted_counts_list:
        proportion = count / total_ques
        sorted_doctype_stats[doctype_name] = {
            'ques_count': count,
            'proportion': proportion,
            'percentage': f"{proportion:.2%}"
        }

    return sorted_doctype_stats


def group_doctypes_and_calculate_unique_categories(data, sorted_doctypes):
    """
    Groups doctypes and calculates the number of unique categories covered by each group.

    Args:
        data (dict): A dictionary with question names as keys and lists of unique doctypes as values.

    Returns:
        dict: A dictionary where keys are tuples of grouped doctypes and values are the number of unique
              questions covered by that group.
    """
    grouped_doctype_coverage = []

    # Iterate to create groups of increasing size, starting from 2
    for i in range(2, len(sorted_doctypes) + 1):
        # Create a group by taking the top 'i' doctypes from the sorted list
        doctype_group = sorted_doctypes[:i]
        covered_ques = set()

        for ques, doctypes_in_ques in data.items():
            # Check if any doctype from the current group is present in the question's doctypes list
            if any(doctype_name in doctypes_in_ques for doctype_name in doctype_group):
                covered_ques.add(ques)

        proportion = len(covered_ques) / len(data)
        grouped_doctype_coverage.append(
            {"group": doctype_group,
             'ques_count': len(covered_ques),
             'proportion': proportion,
             'percentage': f"{proportion:.2%}"
             }
        )

    return grouped_doctype_coverage


def doctype_freq_analysis(doctype_temp_dict):
    # doctypes analysis
    data = {ques: recordtype_dict['record_types'] for ques, recordtype_dict in doctype_temp_dict.items()}
    sorted_doctype_stats = doctype_freq(data)
    sorted_doctypes = list(sorted_doctype_stats)

    grouped_doctype_coverage = group_doctypes_and_calculate_unique_categories(data, sorted_doctypes)

    doctype_analysis_results = {}
    doctype_analysis_results['individual_doctype_metrics'] = sorted_doctype_stats
    doctype_analysis_results['grouped_doctype_coverage'] = grouped_doctype_coverage

    return doctype_analysis_results


def map_norm_class_to_record_type(normalized_doctypes, doctype_temp_results):
    mapped_doctype_temp_results = {}
    for ques_rt_temp_dict in doctype_temp_results:
        if ques_rt_temp_dict["is_processable"] == True:
            mapped_rt = []
            for rt in ques_rt_temp_dict['record_types']:

                unknown_rt = True
                for norm_class, rt_list in normalized_doctypes.items():
                    if ' '.join(rt.split('_')) in rt_list:
                        mapped_rt.append(norm_class)
                        unknown_rt = False
                        break
                if unknown_rt:
                    raise ValueError("unknown record type")
            mapped_doctype_temp_results[ques_rt_temp_dict["query"]] = {'record_types': list(set(mapped_rt)),
                                                                       'temporal_signals': ques_rt_temp_dict[
                                                                           'temporal_signals']}

    return mapped_doctype_temp_results


def map_norm_class_to_temporal(doctype_temp_dict, normalized_temporals):
    mapped_doctype_temp_dict={}
    for doctype, temporals in doctype_temp_dict.items():
        temp=[]
        for temporal in temporals:
            if not re.findall(r"(\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})", temporal) and temporal not in ['acute', 'established', 'chronic', 'resolved', 'initial']:
                check = [norm_class for norm_class, rt_temporal in normalized_temporals.items() if ' '.join(temporal.split('_')) in rt_temporal]
                if check:
                    temp.append(check[0])

        mapped_doctype_temp_dict[doctype] = temp

    return mapped_doctype_temp_dict

query_results_dict = read_json_from_gcs(project_id, nb_results_json, bucket)

doctype_temp_results = deepcopy(query_results_dict)

mapped_doctype_temp_results = map_norm_class_to_record_type(loinc_class_doctype_map_dict, doctype_temp_results)

mapped_doctype_analysis_results = doctype_freq_analysis(mapped_doctype_temp_results)

output_path = f"./{shopping_list_team}/normalized_rst_crs_doctypes_freq_analysis.json"

with open(output_path, "w") as f:
    json.dump(mapped_doctype_analysis_results, f)
