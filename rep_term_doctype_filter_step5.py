from config import project_id, location, gemini_model, nb_results_json, bucket, shopping_list_team
import json
import itertools
from google import genai
from copy import deepcopy
from utils import read_json_from_gcs

#### Dependencies to run this file #####
mapped_doctype_temp_results = "YOUR mapped_doctype_temp_results from running doc_freq_analysis_step3.py"
doctype_to_temp_sum_dict = "YOUR doctype_to_temp_sum_dict from running normalized_temporal_summary_step4.py"
#### End of dependencies to run this file #####


def create_repterm_payload(batch_terms_list):
    prompt = f"""
                I have a list of dictionaries input provided between <list_input> and </list_input> where each dictionary of the list has query_id, query, rep_terms, and doctype_rep_mix keys. query_id is a unique id for each question,
                query is a string of actual query, rep_terms contains list of representative term(s) extracted from the query, doctype_rep_mix contains
                list of tuples where each tuple has two elements where the first element is a document type and the second element is a representative term.
                <list_input> {batch_terms_list} </list_input>.

                For your task, you are performing a two-step contextual tagging and mapping process on the provided input to provide a final result. 
                This means you will process STEP 1 and use its results to proceed to STEP 2 and then give a final result:

                # STEP 1: Context Filtering

                Your Role: You are an expert data processor tasked with creating a clean, high-level indexing system for documents.

                Your Task: You will receive a question and a list of rep_terms. Your job is to decide which terms to keep and which to discard based on a single core principle. The goal is to produce a concise list of high-level "tags" for a document.

                ## The Core Principle: Context vs. Content

                Think of this as sorting files into cabinets. You want to keep the terms that act like cabinet labels (the context) and discard the terms that describe the specific sentences written inside the files (the content).

                1. KEEP terms that provide CONTEXT.
                These are high-level classifiers that describe the type, purpose, status, or structural nature of the information. They answer questions like the following examples:

                What kind of event is this? (e.g., a scan, a referral, a prescription)
                What is the administrative or logistical frame for this information? (e.g., related to insurance, an employer)
                What is the overarching program or status of the patient? (e.g., in hospice, on a clinical trial)

                2. DISCARD terms that are the granular CONTENT.
                These are the specific details, findings, and descriptions that would fill a report. They are often numerous and describe the patient's specific clinical state. This includes but not limited to:

                Specific diagnoses, conditions, and diseases.
                Specific symptoms and clinical findings.
                Specific medication names.
                Specific anatomical parts or lab results that are part of a finding.
                A Simple Litmus Test: Ask yourself, "Is this term a high-level category that defines why a record exists, or is it a specific detail within that record?" Keep the former, discard the latter.

                ## Output of Step 1

                For each item of the input list, keep track of the kept terms and have a brief explanation of your decision for each term, based on the Context vs. Content principle.


                # STEP 2: Semantic Document Type Association

                Your Role: You are now a medical records specialist. Your task is to map a given medical concept to its most appropriate source document type.

                Your Task: Use the list of filtered representative terms of each dictionary, if any, that were kept from Step 1. Use the list of possible pairings called doctype_rep_mix in each dictionary. For each term from the filtered representative terms list of a dictionary, if any, your job is to find the single most logical and specific (document_type, term) pair from the mix in corresponding input dictionary.

                Core Principle: The Primary Source

                You must identify the document type that serves as the primary source or most definitive record for the given term. Ask yourself: "If I needed the official report for this item, what document would I look for?" A term might be mentioned in many documents, but it only has one primary home.

                For example, an MRI result is officially documented in a radiology report, even though doctors might mention the result in their progress_note. Therefore, radiology is the primary source.
                Similarly, a cbc test result is officially documented in a lab_report.

                The final result should be a list that has all the dictionaries of the input and each dictionary must have query_id, its filtered_rep_terms, filtered_doctype_rep_mix, and reasoning.

                ## Examples of Correct Execution

                Example 1:
                {{
                query_id:id_1,
                query: "do you have mri or ct done in past 3 months",
                rep_terms: ["mri", "ct scan"],
                doctype_rep_mix: [(progress_note, mri), (progress_note, ct), (consultation_note, mri), (radiology_report, mri), (radiology_report, ct scan), (radiology_order, mri), (radiology_order, ct scan), (consultation_note, ct) ]
                }}
                Intermediate Step 1 Output:
                {{
                  "reasoning": "'mri' and 'ct scan' describe the type of event. They provide high-level context, not granular content.",
                  "filtered_rep_terms": ["mri", "ct scan"]
                }}
                After Step 2 final result:
                {{
                query_id:id_1,
                filtered_rep_terms: ["mri", "ct scan"],
                filtered_doctype_rep_mix: [(radiology_report, mri), (radiology_report, ct scan), (radiology_order, mri), (radiology_order, ct scan) ],
                "reasoning": "'mri' and 'ct scan' describe the type of event. They provide high-level context, not granular content. The 'radiology' report is the primary source document for both 'mri' and 'ct' procedures. Progress and consultation notes only reference these results."
                }}

                Example 2:
                {{
                query_id:id_2,
                query: "do you have colon cancer",
                rep_terms: ["colon", "cancer"],
                doctype_rep_mix: [(progress_note, colon), (progress_note, cancer), (consultation_note, colon), (radiology_report, cancer), 
                                (radiology_report, colon), (radiology_order, cancer), (radiology_order, colon), (consultation_note, cancer) ]
                }}

                Intermediate Step 1 Output:
                {{
                  "reasoning": "'cancer' and 'colon' describe the granular content of a diagnosis. They are specific clinical details, not a high-level context.",
                  "filtered_rep_terms": []
                }}

                After Step 2 final result:
                {{
                query_id:id_2,
                filtered_rep_terms: [],
                filtered_doctype_rep_mix: [],
                "reasoning":"'cancer' and 'colon' describe the granular content of a diagnosis. They are specific clinical details, not a high-level context.",
                }}


                Example 3:
                {{ query_id:id_3,
                query: "Are you currently in Hospice or has Hospice been recommended?",
                rep_terms: ["hospice"],
                doctype_rep_mix: [(consultation_note, hospice), (hospice_note, hospice), (discharge_summary, hospice), (radiology_report, hospice)]
                }}

                Intermediate Step 1 Output:
                {{
                  "reasoning": "'hospice' is a formal status that defines the entire context of care. It's a high-level classifier.",
                  "filtered_rep_terms": ["hospice"]
                }}

                After Step 2 final result:
                {{
                query_id:id_3,
                filtered_rep_terms: ["hospice"],
                filtered_doctype_rep_mix: [(hospice_note, hospice)],
                "reasoning": "'hospice' is a formal status that defines the entire context of care. It's a high-level classifier.
                            the 'hospice_note' is the most definitive document that outlines the goals and status of hospice care. It is the primary source for this formal status."
                }}


                Example 4 (Crucial Test):
                 {{ query_id:id_4,
                query: "do you have atenolol prescription",
                rep_terms: ["atenolol", "prescription"],
                doctype_rep_mix: [(progress_note, atenolol), (progress_note, prescription), 
                                (lab_report, atenolol), (lab_report, prescription), (consultation_note, prescription), (radiology_report, prescription)]
                }}

                Intermediate Step 1 Output:
                {{
                  "reasoning": "'atenolol' is a specific medication name (content). 'prescription' describes the type of order or event (context).",
                  "filtered_rep_terms": ["prescription"]
                }}

                After Step 2 final result:
                {{
                query_id:id_4,
                filtered_rep_terms: ["prescription"],
                filtered_doctype_rep_mix: []
                "reasoning": "'atenolol' is a specific medication name (content). 'prescription' describes the type of order or event (context). No doctype_rep_mix combination had primary source.",
                }}


                """

    return prompt

# get genai response
def get_genai_repterm_resp(temp_payload):
    client = genai.Client(vertexai=True, project=project_id, location=location)

    response_schema = {
          "type": "object",
          "description": "Container for the list of processed query results.",
          "properties": {
            "results": {
              "type": "array",
              "description": "A list where each item is a dictionary containing the processed output for a single query.",
              "items": {
                "type": "object",
                "description": "The processed output for a single query.",
                "required": [
                  "query_id",
                  "filtered_rep_terms",
                  "filtered_doctype_rep_mix",
                  "reasoning"
                ],
                "properties": {
                  "query_id": {
                    "type": "string",
                    "description": "The unique identifier for the input query."
                  },
                  "filtered_rep_terms": {
                    "type": "array",
                    "description": "The list of selected or filtered representative terms of a query.",
                    "items": {
                      "type": "string"
                    }
                  },
                  "filtered_doctype_rep_mix": {
                    "type": "array",
                    "description": "The final list of (document_type, term) pairs of a query.",
                    "items": {
                      "type": "array",
                      "description": "A pair representing [document_type, term].",
                      "items": {
                        "type": "string"
                      },

                    }
                  },
                  "reasoning": {
                    "type": "string",
                    "description": "A detailed explanation of the logic applied to filter the terms and select the document type pairs."
                  }
                }
              }
            }
          },
          "required": ["results"]
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


# loinc_class_doctype_map_dict


# normalized_temporals
query_results_dict = read_json_from_gcs(project_id, nb_results_json, bucket)
ques_repterm_dict = {temp_dict['query']:temp_dict['representative_terms'] for temp_dict in query_results_dict if temp_dict["is_processable"]==True}

ques_doctype = deepcopy(mapped_doctype_temp_results)

for ques, record_temp_dict in ques_doctype.items():
    record_temp_dict['representative_terms'] = ques_repterm_dict[ques]

# create input for prompt
doctype_terms_list=[]
for i, (ques, doctypes_dict) in enumerate(ques_doctype.items()):
    temp_dict={}
    rep_terms, record_types = doctypes_dict['representative_terms'], doctypes_dict['record_types']

    doctype_rep_mix = list(itertools.product(record_types, rep_terms))
    temp_dict['query_id'] = 'id_'+str(i+1)
    temp_dict['query'] = ques
    temp_dict['rep_terms'] = rep_terms
    temp_dict['doctype_rep_mix'] = doctype_rep_mix
    doctype_terms_list.append(temp_dict)

# split into batches
batch_size=17
batches = [doctype_terms_list[i:i + batch_size] for i in range(0, len(doctype_terms_list), batch_size)]

#
total_results = []
for temp_batch in batches:
    print('new batch')
    temp_payload = create_repterm_payload(temp_batch)
    resp = get_genai_repterm_resp(temp_payload)
    total_results.append(resp['results'])


query_id_to_ques = {ques_info['query_id']: ques_info['query'] for ques_info in doctype_terms_list}

ques_to_filtered_terms={query_id_to_ques[result['query_id']]:result['filtered_rep_terms'] for batch_result in total_results for result in batch_result}

#
doctypes=[]
for ques, doctypes_dict in ques_doctype.items():
    doctypes.extend(doctypes_dict['record_types'])
doctypes = list(set(doctypes))

# create recordtype and their list of representative terms dictionary
doctypes_terms_dict={k: [] for k in doctypes}
for batch_result in total_results:
    for result_dict in batch_result:
        for doctype_term in result_dict['filtered_doctype_rep_mix']:
            if doctype_term[0]=='Note | Emergency department | Document ontology':
                doctypes_terms_dict["Note &#x7C; Emergency department &#x7C; Document ontology"].append(doctype_term[1])
            else:
                doctypes_terms_dict[doctype_term[0]].append(doctype_term[1])

doctypes_terms_dict = {k: list(set(v))for k, v in doctypes_terms_dict.items()}

#doctype_to_temp_sum_dict # normalized_rst_crs_doctype_level_temp_summary.json

# add doctype_terms to doctype_temp_dict
for doctype, temp_dict in doctype_to_temp_sum_dict.items():
    temp_dict['info_in_questions'] = doctypes_terms_dict[doctype]

#doctype_to_temp_sum_dict  # normalized_rst_crs_doctypes_temp_repterms.json


### Change this path according to the task cohort or dt tree etc., ###
output_path = f"./{shopping_list_team}/normalized_rst_crs_doctypes_temp_repterms.json"
with open(output_path, "w") as f:
    json.dump(doctype_to_temp_sum_dict, f)
