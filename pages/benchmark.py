import streamlit as st
import pandas as pd
import base64
import fitz
import os
import json
import time
import filetype
from pydantic import BaseModel, Field

from google.oauth2.service_account import Credentials
from vertexai import init as vertexai_init
from vertexai.generative_models import GenerativeModel, Part

from google import genai
from google.genai import types
import openai
from openai import AzureOpenAI

from dotenv import load_dotenv



# load environment variables
load_dotenv('os.env')

# page config
st.set_page_config(layout="wide")

# caching model resources
@st.cache_resource
def init_vertex_client(model):
    google_credentials = Credentials.from_service_account_info(st.session_state['google_service_account_info'])
    vertexai_init(project=os.environ.get('YOUR_PROJECT'), location="YOUR_LOCATION", credentials=google_credentials)
    model = GenerativeModel(model)
    return model

@st.cache_resource
def init_azure_client():
    api_version = '2024-12-01-preview'
    api_key = base64.b85decode(os.environ.get('YOUR_KEY', '').encode("UTF-8")).decode("UTF-8")
    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint="YOUR_ENDPOINT"
    )


# LLM call cost calculator function
# Google pricing: https://cloud.google.com/vertex-ai/generative-ai/pricing#token-based-pricing
# Azure pricing: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
def llm_call_cost (model, call_metadata):
    cost = 0
    match model:
        case "gemini-2.0-flash-001":
            cost += (call_metadata.prompt_token_count*.13902 + call_metadata.candidates_token_count*0.55608) / 1000000
        case "gemini-2.0-flash-lite-001":
            cost += (call_metadata.prompt_token_count*.06951 + call_metadata.candidates_token_count*0.27804) / 1000000
        case "gemini-1.5-flash-002":
            cost += (call_metadata.prompt_token_count*.000017378 + call_metadata.candidates_token_count*0.00006951) / 1000
        case "gemini-1.5-pro-002":
            cost += (call_metadata.prompt_token_count*.000289625 + call_metadata.candidates_token_count*0.0011585*4) / 1000
        case "p2p-gpt-4o-mini":
            cost += (call_metadata.prompt_tokens*0.15281 + call_metadata.completion_tokens*0.6113) / 1000000
        case "p2p-gpt-4-o":
            cost += (call_metadata.prompt_tokens*2.54677 + call_metadata.completion_tokens*10.1871) / 1000000
    return cost


# process prompt and doc with models
# vertex ai
def vertex_ie (model, prompt, doc, temperature, additional_doc):
    st.session_state['vertexai_generation_config']["temperature"] = temperature
    content = [f'{prompt}']
    if doc:
        # distinguish type of file to use
        file_data = doc.read()
        file_mime = filetype.guess(file_data).mime
        print(f"Document info: {doc.name} {file_mime}")
        document = Part.from_data(
            mime_type=file_mime,
            data=file_data
        )
        content.append(document)
    if additional_doc:
        json_dict = json.loads(additional_doc.read())
        if doc.name in json_dict:
            doc_info = json_dict[doc.name]
            json_str = json.dumps(doc_info)
            content.append(json_str)
        additional_doc.seek(0)
    model_obj = init_vertex_client(model)
    response = model_obj.generate_content(
        content,
        generation_config=st.session_state['vertexai_generation_config'],
        safety_settings=st.session_state['vertexai_safety_settings'],
        stream=False
    )
    cost = llm_call_cost(model, response.usage_metadata)
    return response.text, cost

# azure open ai
def azure_ie (model, prompt, doc, temperature, additional_doc):
    messages = []
    content = [{
        "type": "text",
        "text": prompt
    }]
    if doc:
        # distinguish type of file to use
        file_data = doc.read()
        file_mime = filetype.guess(file_data).mime
        doc.seek(0)
        print(f"Document info: {doc.name} {file_mime}")
        img_bytes = []
        if file_mime == "application/pdf":
            pages_doc = fitz.open(stream=doc.read(), filetype="pdf")
            for page in pages_doc:
                pix = page.get_pixmap()
                img = pix.tobytes("png")
                img_bytes.append(base64.b64encode(img).decode("utf-8"))
        elif file_mime.startswith("image/"):
            pages_doc = fitz.open(stream=doc.read())
            for page in pages_doc:
                pix = page.get_pixmap()
                img = pix.tobytes("png")
                img_bytes.append(base64.b64encode(img).decode("utf-8"))
        for img_byte in img_bytes:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_byte}"
                }
            })
    if additional_doc:
        json_dict = json.loads(additional_doc.read())
        if doc.name in json_dict:
            doc_info = json_dict[doc.name]
            json_str = json.dumps(doc_info)
            content.append({
                "type": "text",
                "text": json_str
            })
        additional_doc.seek(0)
    messages.append({"role": "user", "content": content})

    azure_client = init_azure_client()
    stream = azure_client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        temperature=temperature,
        top_p=0.95,
        max_completion_tokens=8192
    )
    cost = llm_call_cost(model, stream.usage)
    return stream.choices[0].message.content, cost


# Evaluator function for IE task
def evaluate (feature_docs, labels, selected_models, task_prompt, metric_prompt, temperature, additional_doc):
    responses = dict()
    performances = dict()
    ndocs = len(feature_docs)
    
    if not feature_docs or not label_doc:
        return ":red[Error: Either input documents or ground-truth labels' file is not uploaded]", scores

    try:
        for model in selected_models:
            responses[model] = dict()
            performances[model] = dict()
            total_cost = 0
            t0 = time.time()
            for i, doc in enumerate(feature_docs):
                docname = doc.name
                if model.startswith("vertex_ai/"):
                    responses[model][docname], cost = vertex_ie(model.split("/")[-1], task_prompt, doc, temperature, additional_doc)
                    total_cost += cost
                elif model.startswith("azure/"):
                    responses[model][docname], cost = azure_ie(model.split("/")[-1], task_prompt, doc, temperature, additional_doc)
                    total_cost += cost
                else:
                    responses[model][docname] = "Model doesn't exist"
                    break
                doc.seek(0)
                # delay for avoiding violation of rate limit
                if i + 1 < len(feature_docs):
                    time.sleep(delay_param)
            t1 = time.time()
            performances[model]['average_runtime'] = (t1 - t0) / ndocs
            performances[model]['score'] = llm_judge(responses[model], labels, metric_prompt)
            performances[model]['average_cost'] = total_cost / ndocs
            label_doc.seek(0)
    except Exception as e:
        return f""":red[Error: Model: {model} | Doc: {docname} | {e}]""", performances
    
    # save responses
    # savetime = time.time()
    # print(f"Saving {savetime}")
    # with open(f"save/bosch_eval_{savetime}.json", "w") as jf:
    #     json.dump(
    #         {
    #             "responses": responses,
    #             "performances": performances,
    #             "judge": judge_llm,
    #             "modifier": prompt_modifier_llm
    #         }
    #         , 
    #         jf
    #     )
    return responses, performances


# llm judging to score prediction
# metric format
class LLMmetric (BaseModel):
    score: float = Field(description="Score form LLM judge", ge=0, le=1)

# scoring function
def llm_judge (actual_ouputs, expected_outputs, metric_prompt):
    final_prompt = f"""
        You are evaluating a large language model's performance on an information extraction task, based on the actual_outputs, expected_outputs and the metric_prompt.
        You will be given samples of actual_outputs, expected_outputs, which represent the sets of actual and expected outputs. 
        actual_outputs and expected_outputs contain a JSON string each where keys are the filenames of documents that have been processed by the model. So comparison
        for shall be made between values for matching keys present in both JSON strings. In other words, the expected output and actual output for a particular file is found
        under the same key in the respective JSON strings.
        The values not necessarily need to be an exact match for a perfect score. The metric_prompt will clarify what makes the actual output a good prediction. Higher score
        indicates more alignment between the actual and expected outputs in terms of the user's scoring criteria.  The score should range from 0 to 1, 0 being the abysmal performance,
        and 1 being perfect. Score each pair of actual output and expected output separately and calculate the average of scores for all pairs for a final score for the model. The
        user will define in the metric_prompt what sort of metric needs to be used for scoring the large language model's actual output when compared with the expected output.
        Now here are the actual_outputs, expected_outputs, and metric_prompt:
        actual_output: {str(actual_ouputs)}
        expected_output: {str(expected_outputs)}
        metric_prompt: {str(metric_prompt)}
    """
    judge_model_name = judge_llm.split("/")[-1]

    if judge_llm.startswith("vertex_ai/"):
        client = genai.Client(api_key=st.session_state['YOUR_KEY_NAME'])
        response = client.models.generate_content(
            model=judge_model_name,
            contents=[
                final_prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LLMmetric
            )
        )
        score_json = response.text
    
    ## structured outputs deprecated for azure?? response_format argument no longer admissible
    elif judge_llm.startswith("azure/"):
        client = init_azure_client()
        response =  client.beta.chat.completions.parse(
            model=judge_model_name,
            messages=[
                {
                    "role": "user",
                    "content": final_prompt
                }
            ],
            response_format=LLMmetric
        )
        score_json = response.choices[0].message.content

    score = json.loads(score_json)['score']
    return score


# generate better task prompt
def modify_task_prompt (prompt, responses, performances, labels):
    modification_prompt = f"""You will be given a task prompt that has been used for information extraction from documents using large language models. A bunch of models were fed a set of
        the same documents along with the prompt for information extraction. You will be given the prompt, predictions, and labels corresponding to that set. You will also be provided performances
        for each model retrieved from their evaluation on the aforementioned set.
        The prompt is provided as a string.
        The predictions are provided as a JSON string with keys as model names, and each value with another JSON structure where the keys are the document names and the values are the corresponding
        predictions. The labels are provided as a JSON string as well with each key holding the document name, and the values being the corresponding label or expected value.
        The performances are provided as a JSON string as well where the keys are the model names and each value is yet another JSON structure, with keys: score, average_runtime
        are average_cost, whose values represent the score of the model, average runtime taken by the model to process each document, and the monetary cost of processing each
        document for the model, respectively.
        Your job is to modify the prompt such that performances for every model would improve. In other words, lower cost and runtime and higher score is to be pursued.
        The prompt to modify is '{prompt}'
        The predictions are '{str(responses)}'
        The performances are '{str(performances)}'
        The labels are '{str(labels)}'
        Return only the modified prompt and nothing else. No explanation necessary on why the prompt was modified the way it was.
    """
    mod_model_name = prompt_modifier_llm.split("/")[-1]

    if prompt_modifier_llm.startswith("vertex_ai/"):
        client = genai.Client(api_key=st.session_state['YOUR_KEY_NAME'])
        response = client.models.generate_content(
            model=mod_model_name,
            contents=modification_prompt
        )
        modified_prompt = response.text
    elif prompt_modifier_llm.startswith("azure/"):
        client = init_azure_client()
        response = client.beta.chat.completions.parse(
            model=mod_model_name,
            messages=[
                {
                    "role": "user",
                    "content": modification_prompt
                }
            ]
        )
        modified_prompt = response.choices[0].message.content
    return modified_prompt


# session states

if 'selected_models' not in st.session_state:
    st.session_state['selected_models'] = []

if 'benchmark_prompts' not in st.session_state:
    st.session_state['benchmark_prompts'] = {
        "task": "",
        "metric": ""
    }

if 'input_temperature' not in st.session_state:
    st.session_state['input_temperature'] = 0.1

if 'backoff_delay' not in st.session_state:
    st.session_state['backoff_delay'] = 0.0

llm_set = [
    "vertex_ai/gemini-2.0-flash-exp",
    "vertex_ai/gemini-2.0-flash"
]
llm_set_2 = [
    "vertex_ai/gemini-2.0-flash-exp",
    "vertex_ai/gemini-2.0-flash",
    "azure/o3-mini",
    "azure/o1"
]
if 'judge_model' not in st.session_state:
    st.session_state['judge_model'] = llm_set[0]

if 'task_prompt_mod_model' not in st.session_state:
    st.session_state['task_prompt_mod_model'] = llm_set_2[0]



# render page

if 'username' not in st.session_state or not st.session_state['username'] or st.session_state['username'] == '':
    st.write("Please log in on the Home page.")
else:
    st.title("LLM Benchmark ⚡")
    st.write(f"Hi {st.session_state['username']}! Use this page to compare different LLMs on your document IE tasks.")

    # model selection
    model_options = [
        "vertex_ai/gemini-2.0-flash-001",
        "vertex_ai/gemini-2.0-flash-lite-001",
        "vertex_ai/gemini-1.5-flash-002",
        "vertex_ai/gemini-1.5-pro-002",
        "azure/p2p-gpt-4o-mini",
        "azure/p2p-gpt-4-o"
    ]
    selected_models = st.multiselect("Select models for comparison", model_options, default=st.session_state["selected_models"])

    # documents and ground truth upload
    st.write("""Upload documents and label file for evaluation. Selected models will be compared on performance on uploaded document-label pairs. Label file must contain
        keys for corresponding document names that are to be evaluated. Make sure not to have duplicate filenames as multiple documents may get the same prediction assigned.
    """)
    fea, lbl = st.columns(2)
    with fea:
        feature_docs = st.file_uploader(label="Upload PDF documents or image files (JPEG, PNG) to be analyzed", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)
    with lbl:
        label_doc = st.file_uploader(label="Upload single JSON file for labels", type="json", accept_multiple_files=False)
    # additional json file upload for additonal LLM steps if necessary
    st.write("""If additonal information needs to be provided to the documents uploaded above, upload a single JSON document (optional) with keys corresponding to the file names for extraction
        as the corresponding values are fed to the LLMs along with the task prompt and documents. Mention how they shall supplement the extraction task also in the task prompt.
    """)
    additional_doc = st.file_uploader(label="(Optional) Upload single JSON file as additional information on uploaded documents", type="json", accept_multiple_files=False)
        
    
    # prompt inputs
    task_prompt = st.text_area(
        "Enter your prompt to describe the desired Information Extraction task:",
        key="task_prompt",
        height=300,
        value=st.session_state["benchmark_prompts"]["task"] # Persist prompt
    )

    metric_prompt = st.text_area(
        "Enter your prompt to describe the scoring criteria:",
        key="metric_prompt",
        height=100,
        value=st.session_state["benchmark_prompts"]["metric"] # Persist prompt
    )
    
    # model temperature input
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=float(st.session_state['input_temperature']), step=0.05)

    # delay time between LLM calls
    delay_param = st.number_input("Delay between each document extraction (seconds)", min_value=0.0, max_value=10.0, value=float(st.session_state['backoff_delay']))

    # select llms for judge and prompt modifer
    judge_llm = st.selectbox("Select judge LLM to score the models under evaluation on the basis of the criteria mentioned in the metric prompt", llm_set)
    prompt_modifier_llm = st.selectbox("Select LLM to modify the task prompt after evaluation for better results", llm_set_2)


    # run evaluation
    if st.button("Run evaluation", key="evaluation"):
        with st.spinner("Evaluating..."):
            doc_key_error = False
            if label_doc:
                lbls = json.load(label_doc)
                docnames = []
                for doc in feature_docs:
                    docnames.append(doc.name)
                try:
                    labels = {n: lbls[n] for n in docnames}
                    st.session_state['labels'] = labels
                except Exception as e:
                    file_data = doc.read()
                    guess_type = filetype.guess(file_data)
                    st.session_state['responses'] = f":red[Error: Labels' file does not have corresponding keys for the uploaded document. {e}]."
                    doc_key_error = True

            if not feature_docs or not label_doc or not selected_models:
                st.session_state['responses'] = ":red[Error: Either documents or labels' file is not uploaded, or no models have been selected for evaluation]"
            elif doc_key_error:
                pass
            else:
                responses, performances = evaluate (feature_docs, labels, selected_models, task_prompt, metric_prompt, temperature, additional_doc)
                modified_task_prompt = modify_task_prompt(task_prompt, responses, performances, labels)
                st.session_state['responses'] = responses
                st.session_state['performances'] = performances
                st.session_state['modified_task_prompt'] = modified_task_prompt

    # results
    if 'responses' in st.session_state and 'performances' in st.session_state and 'modified_task_prompt' in st.session_state:
        if isinstance(st.session_state['responses'], dict):
            st.divider()
            st.markdown("### Result Summary")
            models, scores, runtimes, costs = [], [], [], []
            for mdl in st.session_state['responses'].keys():
                models.append(mdl)
                scores.append(st.session_state['performances'][mdl]['score'])
                runtimes.append(st.session_state['performances'][mdl]['average_runtime'])
                costs.append(st.session_state['performances'][mdl]['average_cost'])
            res_df = pd.DataFrame({
                "Model": models,
                "Score": scores,
                "Runtime per doc. (seconds)": runtimes,
                "Cost per doc. (EUR)": costs
            })
            res_df.set_index("Model", inplace=True)
            res_df = res_df.style.format(precision=6)
            st.table(res_df)
            st.divider()
            st.markdown("### Predictions")
            for model_name, resps in st.session_state['responses'].items():
                with st.expander(f"{model_name}"):
                    for f, res in resps.items():
                        st.divider()
                        st.write(f"#### File:")
                        st.text(f"{f}")
                        st.write(f"#### Prediction:")
                        st.text(f"{res}")
                        st.write(f"#### Ground truth:")
                        st.json(st.session_state['labels'][f])
            
            # modified task prompt
            st.markdown("### Modified task prompt for better results")
            st.write(st.session_state['modified_task_prompt'])
            st.divider()
        else:
            st.write(st.session_state['responses'])


    # preserve state
    st.session_state['selected_models'] = selected_models
    st.session_state['benchmark_prompts']["task"] = task_prompt
    st.session_state['benchmark_prompts']["metric"] = metric_prompt
    st.session_state['input_temperature'] = temperature
    st.session_state['backoff_delay'] = delay_param
    st.session_state['judge_model'] = judge_llm
    st.session_state['task_prompt_mod_model'] = prompt_modifier_llm