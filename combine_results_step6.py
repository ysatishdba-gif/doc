import json
from google.cloud import storage
from utils import get_cui_dict
from config import project_id, bucket, cui_dict_path, shopping_list_team

### Dependencies to run this file ####
doctype_to_temp_sum_dict = "YOUR final doctype_to_temp_sum_dict variable results from rep_term_doctype_filter_step5.py"
mapped_doctype_analysis_results = "YOUR final mapped_doctype_analysis_results variable results from doc_freq_analysis_step3.py"
ques_to_filtered_terms ="YOUR final ques_to_filtered_terms variable from rep_term_doctype_filter_step5.py"
mapped_doctype_temp_results = "YOUR final mapped_doctype_temp_results variable from normalized_temporal_summary_step4.py"
### End of dependencies to run this file ####

cui_dict, name_to_cui = get_cui_dict(project_id, bucket, cui_dict_path)
doctype_temp_dict = doctype_to_temp_sum_dict  # normalized_rst_crs_doctypes_temp_repterms.json
doctype_freq_dict = mapped_doctype_analysis_results #normalized_rst_crs_doctypes_freq_analysis.json

# get first group that has 100% questions coverage
req_doctypes=[]
for groups in doctype_freq_dict['grouped_doctype_coverage']:
    if groups['proportion']==1:
        req_doctypes.extend(groups["group"])
        break

# new method since one document covers all 66 questions; choose document types that have more than 10% coverage at least
req_doctypes = [temp_doctype for temp_doctype, temp_dict in doctype_freq_dict['individual_doctype_metrics'].items() if temp_dict['proportion']>0.1]

adi_prelim_dict={doc_cat: doc_info for doc_cat, doc_info in doctype_temp_dict.items() if doc_cat in req_doctypes}

# mapped_doctype_temp_results
# create final structure as follows
# output structure
#{'questions': {1: {'text': what is whatquestion, 'record_types':[progres note, dischar], 'rep_terms': [xray, discharge]}}, 'analysis_results': adi_prelim_dict}

#ques_to_filtered_terms # normalized_rst_crs_ques_to_filtered_repterms.json


for ques, ques_terms in ques_to_filtered_terms.items():
    temp_dict={}
    temp_dict['info_in_questions'] = ques_terms
    mapped_doctype_temp_results[ques].update(temp_dict)


ques_details_dict = {i+1:{'text':ques, 'record_types':ques_info_dict['record_types'], 'info_in_questions': ques_info_dict['info_in_questions']}
                     for i, (ques, ques_info_dict) in enumerate(mapped_doctype_temp_results.items())}

analysis_results=[]
for doctype, doctype_dict in adi_prelim_dict.items():

    temp_dict={}
    temp_dict['record_type_name'] = doctype
    temp_dict['record_type_code'] = name_to_cui[doctype]

    temp_dict['temporal_summary'] = {"name": doctype_dict['temporal_summary'], "code": name_to_cui[doctype_dict['temporal_summary']]}

    temp_dict['temporal_concepts'] = [{"name": temporals, "code": name_to_cui[temporals]} for temporals in doctype_dict["temporal_values"]]
    temp_dict['info_in_questions'] = [{"name": ques_info, "code": name_to_cui.get(ques_info, None)} for ques_info in
                                      doctype_dict["info_in_questions"]]

    analysis_results.append(temp_dict)

final_adi_prelim_dict={}
final_adi_prelim_dict['analysis_results']=analysis_results
final_adi_prelim_dict['questions']=ques_details_dict

#final_adi_prelim_dict # normalized_rst_crs_eoam_prelim_results.json

####**** ### Change this path according to the task cohort or dt tree etc., ***** ###

# upload json with doctype (from first frequency group that has 100% questions coverage), temporal info, and representative terms info to gcs
source_blob_name=f"adi_shopping_list/{shopping_list_team}/{shopping_list_team}_normalized_rst_crs_eoam_results.json"
# Convert dictionary to JSON string and upload
storage.Client(project=project_id).bucket(bucket).blob(source_blob_name).upload_from_string(
    data=json.dumps(final_adi_prelim_dict),
    content_type='application/json'
)
