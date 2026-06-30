from fuzzywuzzy import fuzz
from google import genai
import collections
from copy import deepcopy
import json
import subprocess
import concurrent.futures
from config import project_id, location, bucket, gemini_model, cui_dict_path, nb_results_json, doctype_norm_code_json, shopping_list_team
from utils import read_json_from_gcs, get_cui_dict

def auth_gcp(gcloud_loc: str):
    '''
    # Authenticate with GCP environment within the program

    :param gcloud_loc: location of gcloud credentials.json file
    '''

    tmp = subprocess.run([gcloud_loc, 'auth', 'application-default', 'login'], stdout=subprocess.PIPE)
    if tmp.returncode == 1:
        raise Exception("GCloud login failed")


### Map doctypes to loinc norms
def get_match_percentages(target, choices):
    results = []
    for choice in choices:

        ratio = fuzz.ratio(target.lower(), choice.lower())
        if ratio > 50:
            results.append((choice, ratio))

    # Sort results so the highest match is at the top
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    return sorted_results


# auth_gcp('../google-cloud-sdk/bin/gcloud')
cui_dict, name_to_cui = get_cui_dict(project_id, bucket, cui_dict_path)


###------------------------------- *** Normalize record types **** -------------------------------------#####

def get_genai_payload(record_type_name, transactions):
    temp_payload = f""" I have a record type name mentioned between <record type name> and <record type name>. 
            I have a list of transactions mentioned between <transactions> and <transactions> where each transaction has two elements. The first element is the id of the document class
            and the second element is the document class name. Your task is to identify a single document class that is the most closely aligned and hence a best choice/match compared to all other document classes
            to the record type name.  ** It is critical to select only one document class ** The output must be a dictionary with the best choice document class id, its name, and the reason
            for choosing that document class.
            <record type name> {record_type_name} <record type name>
            <transactions> {transactions} <transactions>

            * Examples *
            ** input **
            record type name = 'history and physical'
            transactions = [[id_1, 'History and physical note'], [id_2, 'History and physical panel'], ['id_3', 'Targeted history and physical note']]

            ** output **
            {{"selected_document_class_id": 'id_1',
                "selected_document_class_name": 'History and physical note',
                "reasoning": this is the most closest alignment and a best choice with record type name amongst other choices
            }}

            ** input **
            record type name = 'radiology report'
            transactions = [[id_1, 'X-ray report'], [id_2, 'Radiology panel'], ['id_3', 'Radiology studies']]

            ** output **
            {{"selected_document_class_id": 'id_3',
                "selected_document_class_name": 'Radiology studies',
                "reasoning":  this is the most closest alignment and a best choice with record type name amongst other choices
            }}
            """

    return temp_payload


def get_genai_resp(temp_payload):
    response_schema = {
        "type": "object",
        "description": "The processed output for a single record type name.",
        "required": [
            "selected_document_class_id",
            "selected_document_class_name",
            "reasoning"
        ],
        "properties": {
            "selected_document_class_id": {
                "type": "string",
                "description": "unique id of document class"
            },
            "selected_document_class_name": {
                "type": "string",
                "description": "a string of document class name"
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of the choice for selected document class"
            }
        }
    }

    response = client.models.generate_content(
        model=gemini_model,
        contents=temp_payload,
        config={
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        },
    )

    resp_dict = json.loads(response.text)

    return resp_dict


# read nature breakdown results
query_results_dict = read_json_from_gcs(project_id, nb_results_json, bucket)

all_doctypes = [query_dict["record_types"] for query_dict in query_results_dict if query_dict['is_processable'] == True]
doctypes = [doctype for doctype_list in all_doctypes for doctype in doctype_list]
doctypes = list(set(doctypes))

# read doctype related normalized loinc codes
stnd_cncpt_map_dict = read_json_from_gcs(project_id, doctype_norm_code_json, bucket)
loinc_norms = list(stnd_cncpt_map_dict.values())

# Perform direct string matching
doctypes_matches = {' '.join(doctype.split('_')): [] for doctype in doctypes}
for doctype in doctypes:
    doctype = ' '.join(doctype.split('_'))
    # print(doctype)
    matches = get_match_percentages(doctype, loinc_norms)
    doctypes_matches[doctype] = matches

# doctypes_matches = {k: v[:100]for k, v in doctypes_matches.items()}

# check if there is an 100 match
doctype_to_loinc = {' '.join(doctype.split('_')): [] for doctype in doctypes}
for doctype_name in doctypes_matches:
    for doctype_match in doctypes_matches[doctype_name]:
        if doctype_match[1] == 100:
            doctype_to_loinc[doctype_name] = doctype_match[0]
            break

not100_match_rt = {doctype: doctypes_matches[doctype] for doctype, val in doctype_to_loinc.items() if not val}
doctype_to_loinc = {doctype: val for doctype, val in doctype_to_loinc.items() if val}
print('100% match documents', len(doctype_to_loinc))
print('NOT 100% match documents', len(not100_match_rt))

### Identify normalized doctype loinc for the record types that are not 100% string match using LLM
temp_payloads = []
transactions_dict = {}
c = 1
iter_c = 0

doctype_ids_dict = {doctype: [] for doctype in not100_match_rt.keys()}
doctype_loinc_map = {doctype: [] for doctype in not100_match_rt.keys()}

for doctype, doctype_loinc_norms in not100_match_rt.items():
    record_type_name = doctype
    transactions = [['id_' + str(iter_c + i + 1), tup[0]] for i, tup in enumerate(doctype_loinc_norms)]
    temp_ids = {'id_' + str(iter_c + i + 1): tup[0] for i, tup in enumerate(doctype_loinc_norms)}
    transactions_dict.update(temp_ids)
    doctype_ids_dict[doctype] = list(temp_ids.keys())
    temp_payload = get_genai_payload(record_type_name, transactions)
    temp_payloads.append(temp_payload)
    c += 1
    iter_c += len(transactions)

client = genai.Client(vertexai=True, project=project_id, location=location)
# Execute synchronously written functions in parallel threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(get_genai_resp, temp_payloads), total=len(temp_payloads))

for each_resp in results:
    doctype = [k for k, v in doctype_ids_dict.items() if each_resp['selected_document_class_id'] in v][0]
    doctype_loinc_map[doctype] = transactions_dict[each_resp['selected_document_class_id']]

# add 100% string match record types to the llm identified results
doctype_loinc_map.update(doctype_to_loinc)

# restructure doctype_loinc_map to loinc_class: related_doctype dictionary
loinc_class_doctype_map_dict = collections.defaultdict(list)

for doctype, loinc_class in doctype_loinc_map.items():
    # Append the current 'cui' to the list associated with that 'other_code'.
    loinc_class_doctype_map_dict[loinc_class].append(doctype)

normalized_record_types = dict(loinc_class_doctype_map_dict)
normalized_record_types_cp = deepcopy(normalized_record_types)

loinc_cui_doctypes = {name_to_cui[loinc_class]: doctypes for loinc_class, doctypes in normalized_record_types.items()}

### ----------- PERFORM SECOND PASS LOINC ----------------####
# some normalized loinc class types identified above can be merged together such radiology and imaging can be merged together.
# SECOND PASS logic does that merging.

normalized_record_types = deepcopy(normalized_record_types_cp)
normalized_record_types = {'id_' + str(i + 1): {k: vals} for i, (k, vals) in enumerate(normalized_record_types.items())}

merge_classes_prompt = f"""
                I have a dictionary where each key is a document category id number and its value is another sub-dictionary that has a document category as the key and its value is a list of examples of document types for that category.
            Your task is to analyze this dictionary and generate a 'merge map' in JSON format, identifying categories that should be merged into a single, more comprehensive category.

            CRITICAL MERGING PRINCIPLES:

            Prioritize Clinical and Functional Specificity: Merges MUST be based on a shared, specific clinical domain (e.g., Cardiology, Neurology) or a distinct workflow step (e.g., Patient Admission, Surgical Procedure).
            General Terms Are Insufficient: Do NOT merge categories just because they share general terms like 'note', 'report', 'summary', or 'therapy'. The primary subject matter must be the same. For example, a note about the heart (Cardiology) is completely different from a note about the lungs (Respiratory).
            Distinguish Clinical Domains from Administrative Routes: 'Parenteral' describes a route of administration (by injection/IV), while 'Respiratory' describes a clinical specialty (the lungs). These are fundamentally different and MUST NOT be merged.
            Instructions for Merging:

            Group by Topic: Merge categories that relate to the same overarching clinical concept or workflow, as defined by the principles above.
            Analyze the Full Context: Consider both the category name and their example document types to determine the underlying topic.
            Output Format: Produce a JSON object where the key is the target category id number (chosen from the input) and the value is a list of the other id numbers to be merged into it. Do not create new category id numbers. 
            Avoid: Do not repeat the same category id number in multiple target categories. You should pick only. For Example: Let's there are three categories, id_x, id_y, id_z. If id_x, id_y are both identified as target categories for id_z, you should still assign id_z to only one most relevant target category.

            Example of Correct Merging:

            Categories like 'radiology report' and 'imaging studies' should be merged.
            Reason: Both fall under the broad functional topic of 'Medical Imaging', even if one is a report and the other refers to procedures.
            CRITICAL: Examples of Incorrect Merging:

            'Respiratory therapy note' MUST NOT be merged with 'Parenteral therapy note'.
            Reason: These address entirely different clinical areas. 'Respiratory' pertains to the pulmonary system, while 'Parenteral' describes a non-digestive route of administration. The shared word 'therapy' is irrelevant because the domains are distinct.
            'Patient Intake' MUST NOT be merged with 'Surgical operation note'.
            Reason: These are distinct stages of the patient journey and workflow (administrative check-in vs. a major medical procedure).
            Now, using these strict principles, please analyze the following dictionary and generate the JSON merge map:

             {normalized_record_types}

            """


def get_genai_second_pass(merge_classes_prompt):
    client = genai.Client(vertexai=True, project=project_id, location=location)
    response_schema = {
        "type": "object",
        "title": "Document Category Merge Map",
        "description": "A map where each key is a primary document category id number and its value is a list of sub-categories id numbers to be merged into it.",
        "additionalProperties": {
            "type": "array",
            "description": "A list of one or more category id numbers to merge into the primary category.",
            "items": {
                "type": "string"
            },
            "minItems": 1
        }
    }

    response = client.models.generate_content(
        model=gemini_model,
        contents=merge_classes_prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        },
    )

    resp_dict = json.loads(response.text)

    return resp_dict


resp = get_genai_second_pass(merge_classes_prompt)

# with id numbers in resp
final_dict = {}
avail_classes_llm = []
for record_type_id, syn_rt_id_list in resp.items():
    record_type = list(normalized_record_types[record_type_id].keys())[0]
    avail_classes_llm.append(record_type)
    avail_classes_llm.extend([list(normalized_record_types[syn_rt_id].keys())[0] for syn_rt_id in syn_rt_id_list])
    temp = [doc_name for syn_doc_cat_id in syn_rt_id_list for doc_names_list in
            list(normalized_record_types[syn_doc_cat_id].values()) for doc_name in doc_names_list]
    final_dict[record_type] = temp + [doc_name for doc_names_list in
                                      list(normalized_record_types[record_type_id].values()) for doc_name in
                                      doc_names_list]

for record_type, doc_names_list in normalized_record_types_cp.items():
    if record_type not in avail_classes_llm:
        final_dict[record_type] = doc_names_list

loinc_class_doctype_map_dict = final_dict  # this is second_pass_loinc_class_to_doctypes_map_dict
loinc_cui_doctypes = {name_to_cui[loinc_class]: doctypes for loinc_class, doctypes in
                      loinc_class_doctype_map_dict.items()}

# save the results
output_path = f'./{shopping_list_team}/second_pass_loinc_class_to_doctypes_map_dict.json'
with open(output_path, "w") as f:
    json.dump(loinc_class_doctype_map_dict, f)
