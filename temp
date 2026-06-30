from config import project_id, location, bucket, gemini_model, cui_dict_path, nb_results_json, temporal_norm_code_json, shopping_list_team
from utils import read_json_from_gcs, get_cui_dict
import concurrent.futures
from google import genai
import collections
import json
import re

# load cui dict, input data, temporal norm codes
cui_dict, name_to_cui = get_cui_dict(project_id, bucket, cui_dict_path)
query_results_dict = read_json_from_gcs(project_id, nb_results_json, bucket)
temporal_cui_names_dict = read_json_from_gcs(project_id, temporal_norm_code_json, bucket)

# get cuis of temporal normalized loinc related codes
temporal_cui_names=[]
temporal_cui_names.extend(list(temporal_cui_names_dict['loinc_related_temporal_codes'].values()))
#temporal_cui_names.extend(list(temporal_cui_names_dict['umls_related_temporal_concept_codes'].values()))
temporal_cui_names=list(set(temporal_cui_names))

# read nature breakdown results temporal values
all_temporal_values = [query_dict["temporal_signals"] for query_dict in query_results_dict if query_dict['is_processable']==True]
temporal_values = [temporal for temporal_list in all_temporal_values for temporal in temporal_list]
temporal_values= list(set(temporal_values))
temporal_values = [temporal for temporal in temporal_values if  not re.findall(r"(\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})", temporal) and temporal not in ['YYYY-MM-DD to YYYY-MM-DD', 'acute', 'established', 'chronic', 'resolved', 'initial']]

# find max length temporal value
max_temp_len = {t: len(t) for t in temporal_values}
max_temp_len = sorted(max_temp_len.items(), key=lambda item: item[1], reverse=True)[0][1]

# keep normalized temporal loinc codes that have length less than or equal to max length of nature breakdown temporal value results
temporal_cui_names=[t for t in temporal_cui_names if len(str(t))<=max_temp_len]

# without fuzzy match as first pass - using all temporal concepts of UMLS and not just LOINC - Comment this when using only temporal  loinc codes
temporal_matches = {' '.join(temporal.split('_')): [] for temporal in temporal_values}
for temporal in temporal_values:
    temporal = ' '.join(temporal.split('_'))
    temporal_matches[temporal] = [(norm, 0) for norm in temporal_cui_names]

# without fuzzy match temporal_to_loinc will be empty; so this will just be a placeholder
temporal_to_loinc = {' '.join(temporal.split('_')):[] for temporal in temporal_values}
for temporal_name in temporal_matches:
    for temporal_match in temporal_matches[temporal_name]:
        if temporal_match[1]==100:
            temporal_to_loinc[temporal_name]=temporal_match[0]
            break

not100_match_temporal = {temporal:temporal_matches[temporal] for temporal, val in temporal_to_loinc.items() if not val}
temporal_to_loinc = {temporal:val for temporal, val in temporal_to_loinc.items() if val}


def get_genai_temporal_payload(temporal_type_name, transactions):
    temp_payload = f""" I have a temporal type name mentioned between <temporal type name> and <temporal type name>. 
            I have a list of transactions mentioned between <transactions> and <transactions> where each transaction has two elements. The first element is the id of the temporal class
            and the second element is the temporal class name. Your task is to identify a single temporal class that is the most closely aligned and hence a best choice/match compared to all other temporal classes
            to the temporal type name. ** It is critical to select only one temporal class and if no temporal class is considered to be the best choice then return None ** The output must be a dictionary with the best choice temporal class id, its name, and the reason
            for choosing that temporal class.
            <temporal type name> {temporal_type_name} <temporal type name>
            <transactions> {transactions} <transactions>

            * Examples *
            ** input **
            temporal type name = 'historical'
            transactions = [[id_1, 'Historical'], [id_2, 'Histological type'], ['id_3', 'history']]

            ** output **
            {{"selected_temporal_class_id": 'id_1',
                "selected_temporal_class_name": 'Historical',
                "reasoning": this is the most closest alignment and a best choice with temporal type name amongst other choices
            }}

            ** input **
            temporal type name = 'full history'
            transactions = [[id_1, 'Occupation history'], [id_2, 'disease history'], ['id_3', 'full time history']]

            ** output **
            {{"selected_temporal_class_id": 'id_3',
                "selected_temporal_class_name": 'full time history',
                "reasoning":  this is the most closest alignment and a best choice with temporal type name amongst other choices
            }}

            ** input **
            temporal type name = 'most recent'
            transactions = [[id_1, 'Most recent treponemal test type'], [id_2, 'most recent pregnancy'], ['id_3', 'pre most recent time']]

            ** output **
            {{"selected_temporal_class_id": None,
                "selected_temporal_class_name": None,
                "reasoning":  None of the temporal classes are best choice. 'Most recent treponemal test type' is too specific whereas 'most recent' is general and hence it is also not a best choice.
            }}

            """

    return temp_payload


def get_genai_temporal_resp(temporal_payload):

    response_schema = {
        "type": "object",
        "description": "The processed output for a single temporal type name.",
        "required": [
            "selected_temporal_class_id",
            "selected_temporal_class_name",
            "reasoning"
        ],
        "properties": {
            "selected_temporal_class_id": {
                "type": "string",
                "description": "unique id of temporal class"
            },
            "selected_temporal_class_name": {
                "type": "string",
                "description": "a string of temporal class name"
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of the choice for selected temporal class"
            }
        }
    }

    response = client.models.generate_content(
        model=gemini_model,
        contents=temporal_payload,
        config={
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        },
    )

    resp_dict = json.loads(response.text)

    return resp_dict


temporal_payloads=[]
transactions_dict={}
c=1
iter_c=0

temporal_ids_dict = {temporal: [] for temporal in not100_match_temporal.keys()}
temporal_loinc_map = {temporal: [] for temporal in not100_match_temporal.keys()}

# identify best match temporal normalized loinc concept to a temporal value using LLM
for temporal, temporal_loinc_norms in not100_match_temporal.items():
    temporal_type_name = temporal
    transactions = [['id_'+str(iter_c+i+1), tup[0]] for i, tup in enumerate(temporal_loinc_norms)]
    temp_ids = {'id_' + str(iter_c+i + 1): tup[0] for i, tup in enumerate(temporal_loinc_norms)}
    transactions_dict.update(temp_ids)
    temporal_ids_dict[temporal] = list(temp_ids.keys())
    temporal_payload = get_genai_temporal_payload(temporal_type_name, transactions)
    temporal_payloads.append(temporal_payload)
    c+=1
    iter_c+=len(transactions)

client = genai.Client(vertexai=True, project=project_id, location=location)
# Execute synchronously written functions in parallel threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(get_genai_temporal_resp, temporal_payloads), total=len(temporal_payloads))

# results=[]
# for temporal_payload in tqdm(temporal_payloads):
#     resp = get_genai_temporal_resp(temporal_payload)
#     results.append(resp)


for each_resp in results:
    if each_resp['selected_temporal_class_id']!='None':
        temporal = [k for k, v in temporal_ids_dict.items() if each_resp['selected_temporal_class_id'] in v][0]
        temporal_loinc_map[temporal] = transactions_dict[each_resp['selected_temporal_class_id']]

#check for missing in umls cuis not just loinc cuis i.e., if above LLM cannot find a match for a temporal value then look amongst all UMLS temporal semantic type concepts
temporal_cui_names = temporal_cui_names_dict['umls_related_temporal_concept_codes'].values()
temporal_cui_names=list(set(temporal_cui_names))

missing_temporals = {temporal: [] for temporal, v in  temporal_loinc_map.items() if not v}
max_temp_len = {t: len(t) for t in missing_temporals.keys()}
max_temp_len = sorted(max_temp_len.items(), key=lambda item: item[1], reverse=True)[0][1]
print(max_temp_len)
print(len(missing_temporals))
temporal_cui_names=[t for t in temporal_cui_names if len(str(t))<=max_temp_len]
print(len(temporal_cui_names))

# identify best match temporal umls concept to a unmapped temporal value using LLM
umls_cui_temporals = [(norm, 0) for norm in temporal_cui_names]

for missing_key in missing_temporals.keys():
    temporal_type_name =  missing_key
    transactions = [['id_'+str(i+1), tup[0]] for i, tup in enumerate(umls_cui_temporals)]
    transactions_dict={'id_' + str(i + 1): tup[0] for i, tup in enumerate(umls_cui_temporals)}
    print(len(transactions_dict))
    temporal_payload = get_genai_temporal_payload(temporal_type_name, transactions)
    resp = get_genai_temporal_resp(temporal_payload)
    if resp['selected_temporal_class_id']!='None':
        temporal_loinc_map[temporal_type_name] = transactions_dict[resp['selected_temporal_class_id']]

temporal_loinc_map = {k:v for k, v in temporal_loinc_map.items() if v}

# combine all temporal values and their mappings into single dictionary
temporal_loinc_map.update(temporal_to_loinc)
temporal_loinc_map = {temporal: (norm_class if norm_class else temporal) for temporal, norm_class in temporal_loinc_map.items()}


# create a temporal normalized concept: temporal value dictionary
loinc_class_temporal_map_dict = collections.defaultdict(list)

for temporal, loinc_class in temporal_loinc_map.items():
    # Append the current 'cui' to the list associated with that 'other_code'.
    loinc_class_temporal_map_dict[loinc_class].append(temporal)

loinc_class_temporal_map_dict = dict(loinc_class_temporal_map_dict)


loinc_cui_temporals = {name_to_cui.get(loinc_class, loinc_class):temporals
                       for loinc_class, temporals in loinc_class_temporal_map_dict.items()}

### Change this path according to the task cohort or dt tree etc., and save the results ###
output_path = f'./{shopping_list_team}/cui_to_temporal_map_dict.json'
with open(output_path, "w") as f:
    json.dump(loinc_class_temporal_map_dict, f)

