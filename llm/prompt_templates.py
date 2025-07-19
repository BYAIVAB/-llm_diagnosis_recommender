class PromptTemplates:
    DIAGNOSIS_PROMPT = """
    Analyze the following sensor anomaly data and provide a likely diagnosis.

    Sensor Anomaly Data:
    {anomaly_data}

    Additional Context:
    {context}

    What are the most probable causes for these anomalies?
    """

    RECOMMENDATION_PROMPT = """
    Based on the following diagnosis, provide actionable steps to fix the issue.

    Diagnosis:
    {diagnosis}

    What are the recommended steps to resolve this problem?
    """

    REPORT_PROMPT = """
    Generate a technician-friendly report summarizing the anomaly detection, diagnosis, and recommended actions.

    Anomaly Details:
    {anomaly_details}

    Diagnosis:
    {diagnosis}

    Recommended Actions:
    {recommendations}

    Please provide a concise report, highlighting key findings and actionable insights.
    """
