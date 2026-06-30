from google import genai
import json
from google.cloud import storage
import pandas as pd

''' This logic filters out temporal related codes from the normalized LOINC codes
    And adds temporal_concept semantic type UMLS cuis as well in the final result dictionary
'''


def create_payload(batch_terms_dict):
    ''' prompt to select temporal codes from all normalized concepts'''
    prompt = f"""
                I have a dictionary input provided between <dict_input> and </dict_input> where each key is a concept id and its corresponding value is its concept name. concept id is a unique id for each concept, and
                concept name is a string of actual concept.
                <dict_input> {batch_terms_dict} </dict_input>.
                ** Your task **
                For each concept in the provided dictionary, determine if it can be represent as a generalized or abstract temporal concept or period. This includes descriptions of indefinite durations, broad temporal ranges, general states of recency, historical context, or the current moment etc., provided they are not tied to a specific event or calendar marker. 
                Explain your reasoning for each temporal concept.
                
                Include concept IDs where the description refers to:
                A broad, non-specific period or duration (e.g., '1 to 4 years', 'Three to Five Years', 'Several months', 'Long term', 'Short period').
                A general temporal state (e.g., 'Most Recent', 'Recent', 'Current', 'Historical', 'Past'). The entire concept description should ONLY represent temporal NOT just one or few words in the entire description.
                
                Exclude concept IDs which meet any of the following 3 criterias:
                1) It is *critical* to note that the WHOLE concept description, NOT just one or few words in the description, should DIRECTLY AND ONLY represent a temporal concept. For example, 'Revised - a change in client-provided information' has 'Revised' temporal word in it but it is NOT a temporal concept as the whole description points to additional non-temporal information i.e., client-provided information whereas '1 to 4 years' is a temporal concept since the whole description represents directly and only temporal concept.
                2) A specific, precise date, day, or calendar-defined interval (e.g., 'July 5th', 'Third trimester').
                3) The exact date or time associated with a particular event, observation, question or measurement, even if the description uses terms like 'recent' or 'historical' alongside it (e.g., 'Date Time of Most Recent Angina Pectoris', 'Are you currently using steroid eye drops', 'Date of cardiac diagnosis'). OR

                It is critical that if concepts are not temporal related then return empty results, do not return any irrelevant results.
                
                The output should contain the list of concept ids identified/selected and the reason for selection.
                
                ## Examples
                Input: {{
                'id_1': c,
                'id_2': Three to Five Years,
                'id_3': Liberation day (May 5 every five years),
                'id_4': 'Months 7-9 (third trimester)',
                'id_5': Most Recent,
                'id_6': Date Time of Most Recent Angina Pectoris,
                'id_7': Recent
                'id_8': Established
                'id_9': Chronic
                }}
                
                Output: {{
                id_1: this whole concept description is representing a general time period without any non temporal information 
                 id_2: this whole concept description is representing a  general time period without any non temporal information, 
                 id_5: this whole concept description is representing a general time reference without any non temporal information,
                 id_7: this whole concept description is representing a general time reference without any non temporal information}}
                
                """

    return prompt


# get genai response
def get_genai_resp(temp_payload):
    response_schema = {
            "type": "object",
            "description": "A map of concept IDs to their reasoning as why they are selected as temporal concepts. Each key is a concept ID, and the value is its string reasoning.",
            "additionalProperties": {
                "type": "string",
                "description": "A reasoning of the concept why it is selected as a temporal concept."
            }
        }

    response = client.models.generate_content(
        model=genai_model,
        contents=temp_payload,
        config={
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        },
    )

    resp_dict = json.loads(response.text)

    return resp_dict


def get_cui_dict(project_id, source_bucket):
    storage_client = storage.Client(project=project_id)
    source_bucket = storage_client.bucket(source_bucket)
    blobs = source_bucket.blob("cui_def_dict/cui_definitions.json")
    cui_dict = json.loads(blobs.download_as_string(client=None))

    return cui_dict


def read_json_from_gcs(project_id, source_blob_name, bucket):
    # Initialize the client
    client = storage.Client(project=project_id)

    # Get the bucket and the specific blob (file)
    bucket = client.bucket(bucket)
    blob = bucket.blob(source_blob_name)

    # Download the content as a string
    json_data = blob.download_as_text()

    # Parse the JSON string into a Python dictionary
    return json.loads(json_data)

# set variables
source_bucket = ""
project_id = ""
location = "us-central1"
genai_model = "gemini-2.5-flash"

# all normalized loinc codes
source_blob_name = "Normalized_loinc_classes/norm_code_to_loinc_codes_dict.json"
norm_loincs = read_json_from_gcs(project_id, source_blob_name, source_bucket)

# document type related normalized loinc codes
source_blob_name = "Normalized_loinc_classes/doctype_norm_code_to_loinc_codes_dict.json"
doctype_norm_loincs = read_json_from_gcs(project_id, source_blob_name, source_bucket)

nondoctype_norm_loincs = list(set(set(norm_loincs.keys()) ^ set(doctype_norm_loincs.keys())))

# cui:cui_name dictionary
cui_dict = get_cui_dict(project_id, source_bucket)

# create normalized loinc concept names list
norm_loincs_names = []
for cui in nondoctype_norm_loincs:
    if "_" not in cui:
        if cui in cui_dict:
            norm_loincs_names.append(cui_dict[cui])
        else:
            norm_loincs_names.append("UNKNOWN")
    else:
        cuis = cui.split("_")
        for elem in cuis:
            if elem in cui_dict:
                norm_loincs_names.append(cui_dict[elem])
            else:
                norm_loincs_names.append("UNKNOWN")

# initialize genai client
client = genai.Client(vertexai=True, project=project_id, location=location)

# process normalized loinc codes to identify temporal concepts
all_temporal_loincs=[]
c=0
bs=1000
for i in range(0, len(norm_loincs_names), bs):
    print(i)
    # create a batch of concepts
    batch_terms_list =  norm_loincs_names[i:i+bs]
    batch_terms_dict = {'id_'+str(i+1): name for i, name in enumerate(batch_terms_list)}

    # create payload
    temp_payload = create_payload(batch_terms_dict)
    print('input tokens', len(temp_payload)/4)

    # get genai response for the batch
    resp = get_genai_resp(temp_payload)
    c+=len(resp)
    print('selected count: ', c)
    selected_concepts = [batch_terms_dict[ids] for ids in resp.keys()]
    print(selected_concepts)

    # append batch result to final result list
    all_temporal_loincs.extend(selected_concepts)

# temporal_concept semantic type UMLS cuis
temporal_df= pd.read_csv("gs:/temporal_concept_sem_cuis_in_umls.csv")
umls_temporal_cuis = dict(zip(temporal_df['cui'], temporal_df['cui_name']))

name_to_cui = {name: cui for cui, name in cui_dict.items()}

# Create temporal cui: cui_name dictionary
all_temporal_loincs_cuis_dict={}
for name in all_temporal_loincs:
    all_temporal_loincs_cuis_dict[name_to_cui[name]]=name

temporal_codes={}
temporal_codes['loinc_related_temporal_codes']=all_temporal_loincs_cuis_dict
temporal_codes['umls_related_temporal_concept_codes'] = umls_temporal_cuis

# save the result
with open("./Normalized_loinc_classes/temporal_norm_code_to_loinc_codes_dict.json", "w") as f:
    json.dump(temporal_codes, f)
