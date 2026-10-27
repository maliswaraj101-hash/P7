# MARA — Multi-Agent Research Assistant

LangGraph pipeline that finds recent arXiv papers on a topic, summarizes them with Groq, critiques the summary, and produces a structured research report.

## Pipeline

```
query → fetch papers → summarize → critique → report
```

1. **paper_info** — Search arXiv for the query, download PDFs, extract text  
2. **summary** — Synthesize a topic overview and paper-wise summaries  
3. **evaluate** — Flag weak evidence, logical inconsistencies, and missing perspectives  
4. **report** — Combine summary + critique into a final research report  

## Setup

```bash
git clone https://github.com/Swaraj07082/AI-Research-Agent.git
cd AI-Research-Agent

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### API key

Copy the example env file and add your [Groq](https://console.groq.com/) API key:

```bash
cp .env.example .env
```

Then edit `.env`:

```
GROQ_API_KEY=your_groq_api_key_here
```

Never commit `.env`. It is listed in `.gitignore`.

In the notebook, load the key before creating the LLM (or rely on your shell environment):

```python
from dotenv import load_dotenv
load_dotenv()
```

## Usage

Open and run `main.ipynb`:

```python
initial_state = {
    "query": "papers about attention mechanism"
}
workflow.invoke(initial_state)
```

Change `query` to any research topic you want analyzed.

## Stack

| Piece | Role |
|-------|------|
| [arXiv API](https://arxiv.org/) | Paper discovery |
| [LangGraph](https://langchain-ai.github.io/langgraph/) | Multi-step agent graph |
| [LangChain](https://www.langchain.com/) + Groq | LLM summarization & critique |
| PyPDF | PDF text extraction |

## Project layout

```
├── main.ipynb          # Full pipeline notebook
├── requirements.txt    # Python dependencies
├── .env.example        # Env template (no secrets)
├── .gitignore
└── README.md
```

## License

Use and modify freely for research and learning.
