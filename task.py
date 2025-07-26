import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

def run_ticket_tagging_task():
    """
    This function executes all steps for Task 5: Auto Tagging Support Tickets.
    """
    
    # --- 1. Setup and Initialization ---
    
    # Load environment variables from .env file to get the API key
    load_dotenv()
    
    # Check if the API key is available
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found. Please create a .env file with your key.")
        return

    # Initialize the Large Language Model (LLM) from Google
    # Temperature is set to 0 for more deterministic and consistent output
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

    # --- 2. Dataset Preparation ---

    # Create a sample DataFrame with support tickets. 
    # In a real-world scenario, this would be loaded from a CSV or database.
    data = {
        'text': [
            "My account is locked and I can't log in.",
            "The new software update is causing my system to crash frequently.",
            "I need to reset my password, but the link you sent is not working.",
            "I'm having trouble with the billing for my monthly subscription.",
            "The mobile app is very slow and unresponsive after the last update.",
            "I was charged twice for my last month's bill. I need a refund.",
            "How do I upgrade my current plan to the premium version?",
            "The main dashboard is not loading any data since this morning."
        ],
        'category': [
            "Account Access",
            "Technical Issue",
            "Account Access",
            "Billing",
            "Technical Issue",
            "Billing",
            "General Inquiry",
            "Technical Issue"
        ]
    }
    df = pd.DataFrame(data)

    # Define the possible categories for classification
    categories = ["Account Access", "Technical Issue", "Billing", "General Inquiry"]
    categories_str = ", ".join(categories)

    print("--- Sample Dataset ---")
    print(df)
    print("\n--- Available Categories ---")
    print(categories_str)

    # --- 3. Zero-Shot Learning ---

    print("\n\n--- 3. Zero-Shot Learning ---")
    print("Classifying tickets without any examples...")

    # Create a prompt template that instructs the LLM on its task
    zero_shot_template = """
    You are an expert at classifying customer support tickets.
    Your task is to analyze the support ticket provided below and assign it to one of the following categories: {categories}

    Support Ticket:
    "{ticket_text}"

    Assigned Category:
    """

    zero_shot_prompt = PromptTemplate(
        input_variables=["categories", "ticket_text"],
        template=zero_shot_template
    )

    # Create a LangChain chain to combine the prompt and the LLM
    zero_shot_chain = LLMChain(llm=llm, prompt=zero_shot_prompt)

    # Process each ticket in the DataFrame
    for index, row in df.iterrows():
        ticket_text = row['text']
        true_category = row['category']
        
        # Run the chain to get the prediction
        prediction = zero_shot_chain.run(categories=categories_str, ticket_text=ticket_text).strip()
        
        print(f"\nTicket: '{ticket_text}'")
        print(f"  -> True Category:    {true_category}")
        print(f"  -> Predicted Category: {prediction}")


    # --- 4. Few-Shot Learning ---
    
    print("\n\n--- 4. Few-Shot Learning ---")
    print("Classifying tickets using a few examples to provide context...")

    # Create a prompt template that includes examples (few-shot)
    few_shot_template = """
    You are an expert at classifying customer support tickets.
    Your task is to analyze the support ticket provided below and assign it to one of the following categories: {categories}

    Here are some examples of correctly classified tickets:

    Example 1:
    Ticket: "I can't access my account, it says my login is invalid."
    Category: Account Access

    Example 2:
    Ticket: "My payment for last month was incorrect, I need a refund."
    Category: Billing

    Example 3:
    Ticket: "The application is freezing every time I try to open it."
    Category: Technical Issue
    
    Now, classify the following ticket:

    Support Ticket:
    "{ticket_text}"

    Assigned Category:
    """

    few_shot_prompt = PromptTemplate(
        input_variables=["categories", "ticket_text"],
        template=few_shot_template
    )

    # Create the few-shot chain
    few_shot_chain = LLMChain(llm=llm, prompt=few_shot_prompt)

    # Process each ticket
    for index, row in df.iterrows():
        ticket_text = row['text']
        true_category = row['category']
        
        prediction = few_shot_chain.run(categories=categories_str, ticket_text=ticket_text).strip()
        
        print(f"\nTicket: '{ticket_text}'")
        print(f"  -> True Category:    {true_category}")
        print(f"  -> Predicted Category: {prediction}")

    # --- 5. Outputting Top 3 Tags with Confidence ---

    print("\n\n--- 5. Outputting Top 3 Probable Tags ---")
    print("Asking the model for the top 3 categories in JSON format...")

    # Create a prompt that asks for a structured JSON output
    top_3_template = """
    You are an expert at classifying support tickets.
    Your task is to analyze the support ticket below and provide the top 3 most likely categories from this list: {categories}.
    Please provide your response as a JSON array of objects, where each object has a "category" and a "confidence" key (a value between 0 and 1).

    Support Ticket:
    "{ticket_text}"

    JSON Output:
    """

    top_3_prompt = PromptTemplate(
        input_variables=["categories", "ticket_text"],
        template=top_3_template
    )

    top_3_chain = LLMChain(llm=llm, prompt=top_3_prompt)

    # Process each ticket and get the top 3 predictions
    for index, row in df.iterrows():
        ticket_text = row['text']
        
        prediction_json = top_3_chain.run(categories=categories_str, ticket_text=ticket_text).strip()
        
        print(f"\nTicket: '{ticket_text}'")
        print(f"  -> Top 3 Predicted Categories:\n{prediction_json}")

# --- Main execution block ---
if __name__ == '__main__':
    run_ticket_tagging_task()