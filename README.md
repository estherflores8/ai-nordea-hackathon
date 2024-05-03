# ai-nordea-hackathon

# Nordea FinanceBot: Your AI-Powered Financial Reports Companion

Welcome to DeaNor, your intelligent assistant for navigating through financial reports with ease! Built during the Nordea Bank Hackathon, our project leverages cutting-edge AI technologies and Amazon Web Services to deliver a seamless Q&A experience tailored specifically for financial data.

## Features

- **Intelligent Q&A:** Ask DeaNor anything about your financial reports, and it will provide accurate and insightful answers in real-time.
- **Amazon Tools Integration:** Utilizing Amazon S3 for data storage, Amazon Bedrock for infrastructure, and AWS Lambda functions for seamless operation.
- **Advanced NLP:** We employ vector embeddings with Titan and the powerful SONET v3 Claude model to understand and process complex financial queries effectively.
- **User-Friendly Interface:** Interact with DeaNor effortlessly through the intuitive Streamlit interface, making financial analysis a breeze for everyone.

## Installation

To get started with DeaNor, follow these simple steps:

1. **Clone the Repository:**

   ```bash
   $ git clone https://github.com/estherflores8/ai-nordea-hackathon.git
   ```

2. **Install Dependencies:**

   ```bash
   $ pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   $ streamlit run bedrock_chatbot.py
   ```

The whole application is runned via AmazonSageMaker.

## Architecture

## Usage

Once the application is up and running, open your browser and navigate to the provided URL. You'll be greeted by the sleek and intuitive Streamlit interface of DeaNor. Simply type your financial queries into the chat window, and let FinanceBot handle the rest!

Here are a few example queries to get you started:

- "How much revenue did Nordea generate in Q2 2023?"
- "What were the operating expenses for the fiscal year 2022 for X company?"

## Future improvements

We are truly interested in improving this application with many new features that will boost financial experience.

- Speech-to-text & text-to-speech (AWS Transcribe & Polly)
- Chart generation
- Real-time embeddings
- Finer grain sourcing

## Data

Data is collected in real time from the webpage: https://mfn.se/all/s. Users can interact with it and ask any required questions about the updated pdfs.
![Data Webpage](images/mfn.jpg)

## Visualization example

![Visualization](/images/dornea.png)

## Contributing

Thank you to AWS and Nordea for this amazing hackathon where we got the improve our knowledge on AI using AWS tools.

## Contact

Got questions, feedback, or suggestions? Feel free to reach out to us

Let's revolutionize financial analysis together.
