# Customized LLM Chatbot
Inspired by ChatGPT and other Large Language Models (LLMs), my initial impression was highly positive. However, after extensive use, it became clear that current LLMs have some limitations: they can be unstable, sometimes provide output in an undesired format, occasionally fabricate information, and often fail to deliver accurate answers. To make a chatbot suitable for real-world applications, such as serving customers at a company, we need a more reliable solution.

The solution is Knowledge-Based Retrieval Augmented Generation (RAG). RAG enhances LLMs by supplying them with relevant information based on stored knowledge, such as a vector store.

Additionally, we aim to create a chatbot that not only converses effectively but also can visualize college-related data, as it is designed for Forward Pathway LLC, an educational consulting company. To achieve this, we decompose our questions into several parts, feeding the LLM specific queries for each part. This approach increases the confidence and accuracy of the final response. This can be accomplished using LangChain, but we will employ LangGraph, an extension of LangChain, to design more complex chains of steps.

For rapid development, we will use Streamlit for the app interface and Matplotlib for data visualization. For data, we will utilize Forward Pathway's existing college database.

## Integrating LangGraph, Custom Knowledge Base Vectorstore, RAG, ChatGPT API, and College Data Visualization
Below is the LangGraph flow chart:

![LangGraph flow chart](GraphFlow.png)

This architecture achieves remarkable results.
