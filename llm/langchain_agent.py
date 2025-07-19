import os
# from langchain.llms import OpenAI # Remove this import
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from llm.prompt_templates import PromptTemplates
from langchain_community.llms import Ollama # New import for Ollama

class LLMAgent:
    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434"):
        # No API key needed for local Ollama
        self.llm = Ollama(model=model_name, base_url=base_url)

        # Diagnosis Chain
        diagnosis_prompt = PromptTemplate(
            input_variables=["anomaly_data", "context"],
            template=PromptTemplates.DIAGNOSIS_PROMPT,
        )
        self.diagnosis_chain = LLMChain(llm=self.llm, prompt=diagnosis_prompt, output_key="diagnosis")

        # Recommendation Chain
        recommendation_prompt = PromptTemplate(
            input_variables=["diagnosis"],
            template=PromptTemplates.RECOMMENDATION_PROMPT,
        )
        self.recommendation_chain = LLMChain(llm=self.llm, prompt=recommendation_prompt, output_key="recommendations")

        # Report Chain
        report_prompt = PromptTemplate(
            input_variables=["anomaly_details", "diagnosis", "recommendations"],
            template=PromptTemplates.REPORT_PROMPT,
        )
        self.report_chain = LLMChain(llm=self.llm, prompt=report_prompt, output_key="report")

        self.overall_chain = SequentialChain(
            chains=[self.diagnosis_chain, self.recommendation_chain, self.report_chain],
            input_variables=["anomaly_data", "context", "anomaly_details"], 
            output_variables=["report"],
            verbose=True 
        )

    def run_diagnosis(self, anomaly_dict: dict) -> str:
        anomaly_data = anomaly_dict.get("anomaly_data", "No anomaly data provided.")
        context = anomaly_dict.get("context", "No additional context.")
        anomaly_details = anomaly_dict.get("anomaly_details", "No anomaly details provided.")

        result = self.overall_chain({
            "anomaly_data": anomaly_data,
            "context": context,
            "anomaly_details": anomaly_details 
        })
        return result["report"]

# Example Usage (for testing purposes)
if __name__ == "__main__":
    

    # Initialize LLMAgent without an API key
    agent = LLMAgent(model_name="mistral") 

    sample_anomaly_data = "Sensor A shows sudden spike, Sensor B has erratic readings."
    sample_context = "System was under heavy load during the anomaly period."
    sample_anomaly_details = "Timestamp: 2023-10-27 10:30:00, Sensor: A, Value: 150.5 (Normal: 10-20)"

    test_anomaly_dict = {
        "anomaly_data": sample_anomaly_data,
        "context": sample_context,
        "anomaly_details": sample_anomaly_details
    }

    print("\n--- Running Diagnosis ---")
    try:
        report = agent.run_diagnosis(test_anomaly_dict)
        print("\n--- Generated Report ---")
        print(report)
    except Exception as e:
        print(f"Error during local LLM inference: {e}. Is Ollama running and the model downloaded?")
