# LangChain Chain Lab

An interactive Streamlit project for learning how common LangChain patterns work. It shows each stage of a chain as it runs and includes standalone command-line examples.

## Included patterns

| Pattern | What it demonstrates |
| --- | --- |
| Simple chain | Prompt -> LLM -> output parser |
| Sequential chain | The output of one chain becomes input for the next |
| Parallel chain | Independent notes and quiz branches run, then merge |
| Conditional chain | Feedback is classified and routed to a positive or negative response |

## Project structure

```text
.
├── streamlit_chain.py          # Interactive Streamlit application
├── chain/
│   ├── simple_chain.py         # CLI simple-chain example
│   ├── sequential_chain.py     # CLI sequential-chain example
│   ├── parallel_chain.py       # CLI parallel-chain example
│   └── conditional_chain.py    # CLI conditional-chain example
├── requirements.txt
└── .env.example                # Safe environment-variable template
```

## Setup

1. Create and activate a virtual environment.

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the dependencies.

   ```powershell
   pip install -r requirements.txt
   ```

3. Create your local environment file from the template.

   ```powershell
   Copy-Item .env.example .env
   ```

4. Put your own credentials in `.env`:

   ```dotenv
   GROQ_API_KEY=your_groq_api_key
   GOOGLE_API_KEY=your_google_ai_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   ```

## Run the Streamlit app

```powershell
streamlit run streamlit_chain.py
```

Open the local URL shown in the terminal, normally `http://localhost:8501`.

## Run individual examples

```powershell
python chain/simple_chain.py
python chain/sequential_chain.py
python chain/parallel_chain.py
python chain/conditional_chain.py
```

## Security

`.env`, `.venv/`, caches, logs, and other local-only files are excluded through `.gitignore`. Do not put API keys directly in source code or commit your `.env` file.

## Add tomorrow's runnable

Add the new runnable inside `chain/` (or another clearly named folder), update this README if needed, then commit and push it:

```powershell
git add chain README.md
git commit -m "Add runnable example"
git push
```
