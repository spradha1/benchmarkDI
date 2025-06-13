# Introduction 
This project serves as the Master Thesis for Sanjiv Pradhanang. Title of the thesis is "Optimizing Document Intelligence with LLMs: Design and Implementation of a Benchmarking Framework". The code in this repository builds a Streamlit application serving as a Benchmarking tool for LLMs on Document Information Extraction tasks. Huge thanks to Robert Bosch GmbH for arranging this opportunity.

The repository is not exactly the one used as it is just a scaffolding. The environment variables, API keys and the files relevant to these need to be created.

# Environment
The Streamlit app builds via python libraries. Thus, either an Anaconda environment or a virtual environment is suggested to install all the necessary libraries. Currently, Python 3.11.11 and the latest version of pip is recommended to be installed for this project. Afterwards, in your preferred terminal, navigate to the root directory of this repo, and use the following command to install all the necessary libraries using pip.

    pip install -r requirements.txt

# Credentials & Data
To align with the codebase, create file 'os.env' in the root directory for credentials from providers to access their repsective LLMs. Name your variables for your API keys as you see in the codebase. Also, for Vertex AI, create a separate file for its credentials 'vertex-ai-cred-file.json' in the root directory. The application will run into an error without these files. The data regarding the evaluation by the benchmark will be stored in the 'save' folder in the main directory which you will have to create.

Besides all of this can be changed in the code as per your preference of LLM providers and directory structure.

# Run Application
To run the streamlit application, navigate to the root directory in your terminal, and run the following command.

    python -m streamlit run app.py

Additional arguments regarding server port and address may be used. Instructions regarding how to use the application will be in the home page.
