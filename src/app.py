import os
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from .vectordb import VectorDB
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_documents(data_dir: str = "data") -> List[Document]:
    """
    Load all UK visa policy PDFs.
    (Used by build_db.py)
    """
    results = []
    
    data_path = data_dir 

    print(f"\n📂 Loading documents from '{data_path}' directory...")
    print("=" * 60)
    
    # Check if data path exists
    if not os.path.isdir(data_path):
        print(f"❌ Error: Data directory not found at '{data_path}'")
        return results

    pdf_files = [f for f in os.listdir(data_path) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"⚠️ No PDF files found in {data_path}")
        return results
    
    for idx, filename in enumerate(pdf_files, 1):
        file_path = os.path.join(data_path, filename)
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            results.extend(docs)
        except Exception as e:
            print(f"   ❌ Error loading {filename}: {e}")

    print("=" * 60)
    return results


class RAGAssistant:
    """
    RAG Assistant that connects to a pre-built vector database.
    """

    def __init__(self, prompt_path: str = "prompts/prompt_template.md"):
        """Initialize the RAG assistant - *without* adding documents."""

        print("\n🚀 Initializing JapaPolicy AI RAG Assistant...")
        print("=" * 60)

        # Initialize LLM
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError("Could not initialize LLM. Check GOOGLE_API_KEY.")
        # Connect to existing VectorDB
        print("🔗 Connecting to existing Vector Database...")

        self.vector_db = VectorDB()
        try:
            stats = self.vector_db.get_collection_stats()
            if stats['total_chunks'] == 0:
                 print("⚠️ Warning: Vector Database collection is empty.")
                 print("   Run the 'build_db.py' script first.")
            else:
                 print(f"✅ Connected to collection '{stats['collection_name']}' with {stats['total_chunks']} chunks.")
        except Exception as e:
            print(f"❌ Error connecting to or verifying VectorDB: {e}")
            print("   Ensure 'build_db.py' has run successfully and './chroma_db' exists.")
            raise

        # Load and set up the prompt template
        prompt_full_path = prompt_path
        self.prompt_template = self._load_prompt_template(prompt_full_path)

        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("=" * 60)
        print("✅ RAG Assistant initialized successfully\n")

    def _initialize_llm(self):
        """Initialize the LLM with Google Gemini."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.5-pro")
            print(f"🤖 Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                api_key=api_key,
                model=model_name,
                temperature=0.0
            )
        else:
            print("❌ GOOGLE_API_KEY not found in environment variables.")
            return None

    def _load_prompt_template(self, path: str):
        """Load RAG prompt template from Markdown file."""
        print(f"📝 Loading prompt template from: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt template not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            template_content = f.read()

        print(f"✅ Loaded prompt template.")
        return ChatPromptTemplate.from_template(template_content)

    def _preprocess_query(self, question: str) -> str:
        """Expand query with common synonyms and related terms."""
        q_lower = question.lower()

        # Define expansions for key terms
        expansions = {
        # People & Relationships
            "spouse": "spouse partner husband wife civil partner unmarried partner family member",
            "dependant": "dependent partner child children family member",
            "partner": "partner spouse husband wife civil partner unmarried partner relationship",
            "child": "child children dependent dependant son daughter minor",

        # Visas & Status
            "visa": "visa entry clearance permission permit leave to remain BRP Biometric Residence Permit status immigration",
            "skilled worker": "Skilled Worker visa Tier 2 sponsored work job occupation",
            "health and care": "Health and Care Worker visa NHS medical professional social care",
            "global talent": "Global Talent visa exceptional talent promise Tier 1 endorsement",
            "graduate": "Graduate visa Graduate route post-study work PSW",
            "student": "Student visa Tier 4 study education university CAS Confirmation Acceptance Studies",
            "ilr": "Indefinite Leave to Remain settlement permanent residence settle",
            "settlement": "settlement Indefinite Leave to Remain ILR permanent residence",
            "brp": "Biometric Residence Permit BRP card identity visa document",
            "status": "status leave permission visa lawful residence immigration",
            "overstay": "overstay illegal expired visa immigration breach unlawful presence",
            "section 3c": "Section 3C leave continuation pending application lawful status extend stay",

        # Application Process
            "apply": "apply application process submit form online extend switch renew",
            "switch": "switch change transfer visa category route application inside UK in-country",
            "extend": "extend renew extension application permission stay longer validity",
            "requirements": "requirements eligibility criteria conditions rules qualifications documents evidence proof needed",
            "documents": "documents evidence proof required needed submit support application",
            "proof": "proof evidence documents bank statements payslips certificate letter",
            "cos": "Certificate of Sponsorship CoS sponsor reference number defined undefined",
            "cas": "Confirmation Acceptance Studies CAS student sponsor university reference",
            "sponsor": "sponsor licence employer company organisation Sponsor Management System SMS A-rated CoS Certificate Sponsorship",
            "fee": "fee cost price payment charge application visa IHS Immigration Health Surcharge priority super",
            "priority service": "priority super priority service fast track decision expedite application fee",
            "time": "time duration period processing timeline validity expiry length grant how long decision wait",
            "refusal": "refused rejected refused application appeal review reconsideration challenge decision grounds",
            "appeal": "appeal review reconsideration challenge refusal decision tribunal",

        # Work & Finance
            "work": "work employment job occupation role profession permitted voluntary unpaid",
            "salary": "salary income earnings pay threshold wage minimum going rate SOC code pay scale band",
            "going rate": "going rate salary minimum pay occupation code SOC threshold",
            "funds": "maintenance funds proof money bank statements savings financial requirement",
            "maintenance": "maintenance funds proof money bank statements savings financial requirement",
            "soc code": "SOC Standard Occupational Classification occupation code job title eligible list",

        # Other Key Terms
            "english language": "English language requirement test B1 A1 CEFR IELTS SELT UKVI approved secure test",
            "ihs": "Immigration Health Surcharge IHS healthcare fee payment NHS",
            "nhs surcharge": "Immigration Health Surcharge IHS healthcare fee payment NHS",
            "tb test": "tuberculosis TB test certificate medical examination clinic approved",
            "absences": "absences continuous residence ILR settlement time outside UK travel",
            "continuous residence": "continuous residence period absences ILR settlement 5 years 10 years",
            "genuine": "genuine relationship genuine intention genuine student genuine vacancy test",
            "immigration rules": "Immigration Rules statement changes HC appendix guidance policy law",
            "statement of changes": "Statement Changes Immigration Rules HC update policy law", # Slightly broader
            "guidance": "guidance policy caseworker rules instructions Home Office UKVI",
        }
        expanded_terms = []
        for term, expansion in expansions.items():
            if term in q_lower:
                expanded_terms.append(expansion)
        if expanded_terms:
            return f"{question} {' '.join(expanded_terms)}"
        return question

    def invoke(self, question: str, n_results: int = 7, use_hybrid: bool = True) -> Dict[str, Any]:
        """
        Query the RAG assistant using the pre-built database.
        """
        if not question or not question.strip():
            return self._empty_response("Please enter a valid question.")

        question = question.strip()

        expanded_query = self._preprocess_query(question)

        try:
            search_results = self.vector_db.search(
                expanded_query,
                n_results=n_results,
                min_similarity=0.65,
                use_hybrid=use_hybrid
            )
        except Exception as e:
             print(f"❌ Error during vector DB search: {e}")
             return self._empty_response("An error occurred while searching the document database.")

        if not search_results or not search_results.get("documents"):
            return self._empty_response(
                "I couldn't find relevant information in the documents. "
                "Please try rephrasing or check GOV.UK."
            )

        similarities = search_results.get("similarities", [])
        if not similarities:
             return self._empty_response("No similarity scores available.")
        
        # Get top semantic similarity score
        top_semantic_similarity = search_results.get("top_semantic_sim", 0.0)
        # Calculate average similarity of the *final returned* chunks
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        # Determine confidence based on top semantic similarity
        if top_semantic_similarity >= 0.85:
            confidence, confidence_emoji = "high", "🟢"
        elif top_semantic_similarity >= 0.75:
            confidence, confidence_emoji = "medium", "🟡"
        else:
            confidence, confidence_emoji = "low", "🔴"

        # Format context for the LLM
        context_parts = []
        sources_used = []
        for i, (doc, meta, sim) in enumerate(zip(
            search_results["documents"],
            search_results["metadatas"],
            similarities
        ), 1):
            source_file = meta.get("source", "unknown")
            page = meta.get("page", "N/A")

            # Include relevance score in context for LLM
            context_parts.append(f"[Source {i}: {source_file}, Page {page}, Relevance: {sim:.1%}]\n{doc}")
            sources_used.append({
                "rank": i, "file": source_file, "page": page,
                "similarity": round(sim, 3),
                "quality": "high" if sim >= 0.85 else "medium" if sim >= 0.75 else "low"
            })

        context = "\n\n---\n\n".join(context_parts)

        # Generate answer using LLM
        try:
            response = self.chain.invoke({"context": context, "question": question})
            answer = response.strip() if response else "No answer generated."

            # Check for generic answers
            generic_phrases = [
                "i couldn't find specific information",
                "i couldn't find relevant information",
                "i couldn't find an exact answer",
                "the provided documents do not contain",
                "based on the provided context",
                "check the latest updates on gov.uk",
                "please refer to gov.uk"
            ]
            is_generic = any(phrase in answer.lower() for phrase in generic_phrases)

            if is_generic and confidence == "high":
                confidence, confidence_emoji = "medium", "🟡"
    
            elif is_generic and confidence == "medium":
                 confidence, confidence_emoji = "low", "🔴" 

            return {
                "answer": answer,
                "sources": sources_used,
                "confidence": confidence,
                "confidence_emoji": confidence_emoji,
                "retrieved_chunks": len(search_results["documents"]),
                "avg_similarity": round(avg_similarity, 3),
                "top_similarity": round(top_semantic_similarity, 3),
                "search_type": search_results.get("search_type", "unknown")
            }

        except Exception as e:
            print(f"❌ Error invoking LLM chain: {e}")
            return self._empty_response("An error occurred while generating the response.")

    def _empty_response(self, message: str) -> Dict[str, Any]:
        """Return standard empty/error response structure."""
        return {
            "answer": message, "sources": [], "confidence": "low",
            "confidence_emoji": "🔴", "retrieved_chunks": 0,
            "avg_similarity": 0.0, "top_similarity": 0.0, "search_type": "none"
        }
