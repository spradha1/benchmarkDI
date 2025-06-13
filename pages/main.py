import streamlit as st
import os
import base64
import logging
import json
import getpass

from vertexai.generative_models import SafetySetting

from dotenv import load_dotenv



# load environment variables
load_dotenv('os.env')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(layout="wide")

st.write(st.__version__)

# cache credentials
username = getpass.getuser()
if username != '':
    with open('vertex-ai-cred-file.json', 'r') as f:
        google_service_account_info = json.load(f)
        st.session_state['google_service_account_info'] = google_service_account_info
        # google credentials
        your_key_1 = base64.b85decode(os.environ.get("YOUR_KEY_1", '').encode("UTF-8")).decode("UTF-8")
        st.session_state['YOUR_KEY_NAME'] = your_key_1
        # azure credentials
        your_key_2 = base64.b85decode(os.environ.get("YOUR_KEY_2", '').encode("UTF-8")).decode("UTF-8")
        st.session_state['YOUR_KEY_NAME'] = your_key_2
    account = {"name": username}
    st.session_state['username'] = account["name"]


if st.session_state['username'] != '':
    st.write("# 🔆 Document Information Extraction Benchmark 🔆")
    st.write(f"Hi {st.session_state['username']}!")

    st.write("## A Benchmark for Custom Dcoument IE Tasks!")
    st.markdown("""
        We present to you an LLM benchmark for whatever custom DI task you may have.
        ### How to use this app?
        **👈 Navigate the app using the sidebar**.
        - **❄️ Home**: This place ⬇️.
        - **⚡ Benchmark**: Have your dataset and prompts ready in the required format for testing out the available LLM models.
        **IMPORTANT** - The app might restart your session if you are idle for too long (e.g. grabbing lunch or coffee). Make sure to save your prompts (Notepad, Word, etc.).

        ### Operating Benchmark Application
        **Input Fields**
        - Models: From a multiselect dropdown, select the LLMs to be compared on your IE task.
        - Documents: Upload multiple PDF/JPG/JPEG/PNG documents on which IE is to be performed.
        - Labels: Upload the ground-truth labels in a single JSON file with keys corresponding to the filenames of the documents uploaded. Thus, every document uploaded must have their filename
            as one of the keys in the labels' file for proper evaluation otherwise the evaluation will result in an error. Values should be the expected information to be extracted. The values may
            be from a single valuto a larger json object representing different pieces of information. It needs to be structured that way as our LLM judge is instructed to reognize pairs of
            prediction and ground-truth in such manner.
        - Additional document: This an optional input if one wants to provide further information relevant to the documents. For convenient parsing, a JSON file with keys corresponding to the
            filenames of the documents uploaded shall be provided. It is optional to have keys for all filenames. This may serve an entirely different purpose other than help with extraction itself.
            For instance, a consequent step of verification on the extracted information. Try to instruct the LLM in the task prompt how this information can supplement your desired task.
        - Task prompt: Write a prompt that best describes your IE task and the output formaexpected. The LLMs under comparison take in the prompt along with the documents tgenerate the output.
        - Metric prompt: Write a prompt that would details the criteria for scoring to btaken into consideration by an LLM judge. If a task requires other than simple exacmatching to measure the
            performance of an LLM, this is the place to describe it. Another LLM takes in the prompt, along with the predictions and the ground-truth labels, and generates score between 0-1 for
            each LLM.
        - Temperature: The "temperature" parameter in LLMs controls the randomness or creativity of the model's output. It's a crucial parameter influencing the balancbetween predictable,
            deterministic text and surprising, potentially nonsensical text. Currently admissible range for temperature is from 0 to 2 for the multimodal LLMs froAzure and Google.
            Think of the temperature like this:
            - **Low Temperature**: The model will be very deterministic and predictable. Iwill select the most likely next word(s) at each step, producing text that igenerally coherent
               and consistent but potentially less creative or diverse. This is useful wheyou need precise, factual answers or consistent style. The output will be more focused and less likely
               to deviate from the expected.
            - **High Temperature**: The model will be more random and creative. It wilconsider a broader range of possible next words, even less likely ones. This calead to more surprising
               and diverse outputs, but also to incoherent or nonsensical text. This is suitable for generating creative text formats like poems, fiction, obrainstorming ideas. The output is
               more unpredictable and might explore less common word choices.
            - **Medium Temperature**: This represents a balance between predictability and creativity. The model will still choose the most likely words but will also introduce some randomness,
               leading to outputs that are generally coherent and interesting. It's a good starting point for many tasks.

        **Output**
        - Result Summary: A table for comparison of all models selected. Contains metrics including the score from the LLM judge, time taken per document, and cost charged by the LLM provider
            per document.
        - Predictions: Then comes a stack of expandable widgets with the model names as the headers. One can view for each document what the model predicted and what was the corresponding
            ground truth.
        - Modified task prompt for better results: The task prompt you provided on the most recent run is fed into another LLM along with the predictions, ground-truth labels, and the metrics,
            to retrieve a better suggestion for the task prompt. You may try this modified prompt in place of the previous one to see if that improves the score, and also analyze what details are
            added or omitted from the previous prompt.

        ### Tips & tricks
        - **Specify clear instructions (Prompt Engineering)**: The quality of your prompt directly impacts the output. Be precise in your instructions, specifying the desired task (e.g., "Summarize
            this legal document," "Extract key entities from this research paper," "Translate this document to Spanish"). Provide context where necessary.
        - **Iterative refinement**: Don't expect perfect results on the first try. Experiment with different prompts and the temperature parameter. Review the output and iterate until you achieve
            the desired outcome.
        - **Try structuring outputs**: For tasks requiring structured data (e.g., extracting information), guide the model to provide data in a structured format. Use json formatting, tables, or
            other formatting cues to help the model understand the desired output.
        - **Leverage context**: Provide relevant context in your prompt to help the model understand the task better. Reference specific sections of the document, provide examples, or offer
            additional information to guide the model.
        - **Handle ambiguity**: GenAI can struggle with ambiguous language. If your document contains vague or unclear phrasing, try to clarify it before processing.
        - **Verify output**: Always manually review the output from GenAI. Don't blindly trust the results, especially for critical tasks. The model may make mistakes, particularly with complex or
            nuanced information.

        ### Troubleshooting
        - **Connection error** - there is an error displayed saying something about a lost connection!
            - **Solution** - Try running the evaluation again. If the problem persists over multiple tries, please contact AAA. 
        - **User error** - there is an error displayed saying something about a user error!
            - **Solution** - Go back to the **main** page using the sidebar. If the problem persists, please contact AAA.
        - **Any other error** - There is any other error message displayed!
            - **Solution** - Just try to do the same thing again. If the problem persists, please contact AAA.
        - **I don't get the outputs from the AI that I want!**
            - **Solution** - Try to refine your prompt 🤷‍♀️.
    """
    )

    # Safety settings
    st.session_state['vertexai_safety_settings'] = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
    ]

    st.session_state['vertexai_generation_config'] = {
        "max_output_tokens": 8192,
        "temperature": 0.1,  # default value, will be updated via slider
        "top_p": 0.95
    }
else:
    st.write("Please log in to access the app.")