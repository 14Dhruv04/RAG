# Agentic RAG Systems
This is a documentation of my learnings about Agentic RAG Systems.

## What is Agentic RAG?
A system that allows the LLM to autonomously plan their next steps while pulling information from either the knowledge base or external sources. The LLMs are provided with ***Tools*** that they can use to retrieve/manipulate data. It involves iterative calls to LLM and after each call the system evaluates the results, refines the queries and, if required, invokes additional tools and continues this cycle until a satisfactory answer/solution is acheived.

- **Tools**: nothing but functions that perform certain specific operations on a given input to enable to LLM perform a specific task and respond accordingly.

## Why Agentic RAG?
A question that I got while diving into the topic was - "Why can't we simply create a while loop over the LLM call and in each iteration, ask the LLM itself to generate and evaluate the answer, if the LLM is not satisfied with the answer, it will say so and present a newer query with which the loop will run again making tool call with the new query?". Well, that is basically an Agent and how it works. What makes it different than the simple while-loop logic is:
- ability to traceback to a previous answer.
- ability to make parallel tool calls or calls to sub-agents with separate contexts
- robust self-correction mechanism


### Tech Stack:
***
- **LangGraph** - for agents
- **LangChain** - for semantic similarity search and document loading/parsing
- **Streamlit** - to create the web application
- **Python** - to code


## My Implementation
I have created a basic Agentic RAG Chatbot that has the ability to parse through .pdf, .xls, .xlsx and .csv files and answer questions in those documents using two types of *retrieval tools*. 

### Retrieval Tools:
- **BM25 Retriever** (```bm25_tool.py```): a totally keyword-based retrieval mechanism built on the concept of TF-IDF. (bag of words)
- **Semantic Retriever** (```semantic_tool.py```): a cosine-similarity based semantic search retriever that enables retrieving the most relevant (to the query) chunks.