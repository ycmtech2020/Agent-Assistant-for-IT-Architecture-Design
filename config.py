import os

# ===========================
# AZURE OPENAI CONFIG
# ===========================

# 🔹 Azure OpenAI endpoint (resource URL)
#    Looks like: "https://<your-resource-name>.openai.azure.com/"
AZURE_OPENAI_ENDPOINT = "https://genailab.tcs.in/"

# 🔹 Azure OpenAI API key
#    This is the key you see in Azure Portal for your OpenAI resource
AZURE_OPENAI_API_KEY = "sk-YIqkij05K0T3OA20M6gtJg"

# 🔹 Deployment name for your GPT-4o (or other) model in Azure
#    This is NOT the base model name, but the deployment name you created.
#    e.g. "genailab-maas-gpt-4o" if that's your deployment.
OPENAI_MODEL = "azure/genailab-maas-gpt-4o"

# Path to the templates JSON file
TEMPLATES_PATH = os.path.join("data", "templates.json")
