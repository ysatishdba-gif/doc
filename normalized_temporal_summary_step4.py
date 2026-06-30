from config import project_id, location, gemini_model, shopping_list_team
import json
from google import genai
from copy import deepcopy
import re

### Dependencies to run this file ######
mapped_doctype_analysis_results = "YOUR mapped_doctype_analysis_results variable from running doc_freq_analysis_step3.py"
mapped_doctype_temp_results = "YOUR mapped_doctype_temp_results variable from running doc_freq_analysis_step3.py"
loinc_class_temporal_map_dict = "YOUR loinc_class_temporal_map_dict from running map_temporal_to_norm_class_step2.py"
### End of dependencies to run this file ####


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

# temporal analysis
def perform_temporal_analysis(temporal_dict):
    temp_summary_prompt = f"""
    You are an expert data analyst specializing in temporal information. Your task is to synthesize and reconcile a list of time-related phrases into a single, concise, human-readable summary.

    Context:
    - The summary should be the most compact representation of the combined phrases. Follow the Priority mentioned below:

    Priority:
    The summary of each list of temporal components should follow the following priority:
    ** First priority **: If temporal components have time range components or time period such as last 3 months, past 6 months etc. Summarization should be done only using these components by taking the longest period.
    ** Second priority **: If first priority does not exist for the list then second priority is to check for latest and future related temporal components such as current, recent, active, future etc. Summarization should be done only using these components.
    ** Third priority **:  If first and second priority is not possible for the list then third priority is to check for historical phrased temporal components such as historical, full history etc. Summarization should be done only using these components.

    * It is critical to note that summarized temporal name should be extracted from the provided input list of each id. DO NOT create your own summarized temporal name. 

    Example:
    Input Data: 
    {{'id1': ['Within 30 days', 'past 6 months', '6 months or less', 'last 3 months', 'current', '2025-10-24 to 2026-04-24'], 'id2': ['last 12 months', 'last 9 months', 'historical'],
        'id3': ['current', 'most recent', 'future', 'historical']}}

    Output: {{'id1': 'last 6 months', 'id2': 'last 12 months', 'id3': 'most recent'}} 

    Instructions:
    I have a dictionary mentioned between <temporal_dict> and </temporal_dict>, where each key is an id and its value is a list of temporal components. 
    Summarize each list of temporal components and return a dictionary output where each key is the id and its value is
    its corresponding temporal components summarized result. if the temporal component list has only one temporal element then return that element as is in the summary.

    <temporal_dict> 
    {temporal_dict}
    </temporal_dict>
    """

    client = genai.Client(vertexai=True, project=project_id, location=location)

    response_schema = {
        "type": "object",
        "description": "A dictionary mapping unique item IDs to their summarized temporal strings.",
        "additionalProperties": {
            "type": "string",
            "description": "The temporal summary for the item ID."
        }
    }

    response = client.models.generate_content(
        model=gemini_model,
        contents=temp_summary_prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        },
    )

    resp_dict = json.loads(response.text)

    return resp_dict


all_doctypes = list(mapped_doctype_analysis_results['individual_doctype_metrics'].keys())

doctype_temp_dict={doctype_name:[] for doctype_name in all_doctypes}
for doctype_name in all_doctypes:
    temporals=[]
    for ques, recordtype_dict in mapped_doctype_temp_results.items():
        if doctype_name in recordtype_dict['record_types']:
            temporals.extend(recordtype_dict['temporal_signals'])

    doctype_temp_dict[doctype_name] = list(set(temporals))

normalized_temporals = deepcopy(loinc_class_temporal_map_dict)
mapped_doctype_temp_dict = map_norm_class_to_temporal(doctype_temp_dict, normalized_temporals)

temporal_dict = {'id_'+str(i+1): temporal for i, (doctype, temporal) in enumerate(mapped_doctype_temp_dict.items())}
temporal_id_ques = {'id_'+str(i+1): doctype for i, (doctype, temporal) in enumerate(mapped_doctype_temp_dict.items())}

resp_dict = perform_temporal_analysis(temporal_dict)

doctype_to_temp_sum_dict={} # normalized_rst_crs_doctype_level_temp_summary.json
for id_k, temp_summary in resp_dict.items():
    doctype_to_temp_sum_dict[temporal_id_ques[id_k]]={
        "temporal_summary": temp_summary,
        "temporal_values": temporal_dict[id_k]
    }


### Change this path according to the task cohort or dt tree etc., ###
output_path = f"./{shopping_list_team}/normalized_rst_crs_doctype_level_temp_summary.json"
with open(output_path, "w") as f:
    json.dump(doctype_to_temp_sum_dict, f)
