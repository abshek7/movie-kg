from os import getenv
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import numpy as np

load_dotenv()

NEO4J_URL = getenv("NEO4J_URL")
NEO4J_USER = getenv("NEO4J_USER")
NEO4J_PASSWORD = getenv("NEO4J_PW")
NEO4J_DATABASE = getenv("NEO4J_DB")
GEMINI_API_KEY = getenv("GEMINI_API_KEY")


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)

@st.cache_resource
def initialize_vector_store():
    # Connect to Neo4j
    graph = Neo4jGraph(
        url=NEO4J_URL,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )
    
    query = """
    MATCH (d:Person)-[:DIRECTED]->(m:Movie)
    RETURN m.title as title, m.year as year, m.rating as rating, 
           collect(d.name) as directors, m.tagline as tagline
    """
    
    results = graph.query(query)
    
    documents = []
    metadatas = []
    
    for result in results:
        # Create a rich text representation for better semantic search
        doc_text = f"Movie: {result['title']} ({result['year']})\n"
        doc_text += f"Directed by: {', '.join(result['directors'])}\n"
        if result['tagline']:
            doc_text += f"Tagline: {result['tagline']}\n"
        doc_text += f"Rating: {result['rating']}"
        
        documents.append(doc_text)
        metadatas.append({
            'title': result['title'],
            'year': result['year'],
            'rating': result['rating'],
            'directors': result['directors'],
            'tagline': result['tagline']
        })
    
    # Create FAISS vector store
    vector_store = FAISS.from_texts(
        documents,
        embeddings,
        metadatas=metadatas
    )
    
    return vector_store, graph

@st.cache_resource
def graph_chain():
    vector_store, graph = initialize_vector_store()
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        google_api_key=GEMINI_API_KEY, 
        temperature=0
    )
    
    chain = GraphCypherQAChain.from_llm(
        graph=graph, 
        llm=llm, 
        return_intermediate_steps=True, 
        verbose=True,
        allow_dangerous_requests=True,
        top_k=5,
        cypher_prompt_template="""You are a helpful assistant that generates Cypher queries for Neo4j.
        Use the following schema to generate valid Cypher queries:
        
        Node properties:
        Movie {title: STRING, year: INTEGER, rating: FLOAT, certificate: STRING, run_time: INTEGER, tagline: STRING}
        Person {name: STRING}
        Genre {name: STRING}
        
        Relationships:
        (:Movie)-[:BELONGS_TO]->(:Genre)
        (:Person)-[:WROTE]->(:Movie)
        (:Person)-[:ACTED_IN]->(:Movie)
        (:Person)-[:DIRECTED]->(:Movie)
        
        Question: {question}
        
        Generate a Cypher query that answers the question. The query should:
        1. Use CONTAINS and toLower() for text matching
        2. Return relevant details (title, year, rating)
        3. Order results by year (newest first)
        4. Return only the Cypher query without any explanation"""
    )
    
    return chain, vector_store


def infer(chain, vector_store, prompt):
    try:
        semantic_results = vector_store.similarity_search(prompt, k=3)
        
        # Then, perform graph query
        response = chain.invoke(prompt)
        query = response["intermediate_steps"][0]["query"]
        context = response["intermediate_steps"][1]["context"]
        result = response["result"]
        
        # Combine results
        combined_result = f"Semantic Search Results:\n"
        for doc in semantic_results:
            combined_result += f"\n{doc.page_content}\n"
        
        combined_result += f"\nGraph Query Results:\n{result}"
        
        return query, context, combined_result
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None, None, "I apologize, but I encountered an error while processing your query. Please try rephrasing your question."


if __name__ == "__main__":
    st.subheader("IMDbot")
    st.write("Made using Neo4j, Langchain, Gemini, and FAISS.")
    st.write("Ask questions about movies, directors, actors, or genres. The search uses both semantic and graph-based matching.")
    
    chain, vector_store = graph_chain()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask questions about IMDb's top 250."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        query, context, result = infer(chain, vector_store, prompt)
        with st.chat_message("assistant"):
            if query:
                st.code(query, language="cypher")
                if context:
                    st.write("Found results:", context)
            st.markdown(result)
        st.session_state.messages.append({"role": "assistant", "content": result})
